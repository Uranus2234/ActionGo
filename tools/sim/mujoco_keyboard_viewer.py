"""GLFW MuJoCo viewer with key press/release and mouse perturbation support."""

import ctypes
import os
import threading
import time

import mujoco
import numpy as np


class KeyboardViewer:
    def __init__(self, model, data, bridge, title="ActionGo Sim2Sim", key_callback=None,
                 width=1920, height=1080, asynchronous=False, render_backend="glfw"):
        import glfw

        self.glfw = glfw
        self.model, self.data, self.bridge = model, data, bridge
        self.state_lock = getattr(bridge, "state_lock", threading.RLock())
        self.egl_context = None
        self.egl = None
        self.gl = None
        self.renderer_name = ""
        self.backend_warning = ""
        if render_backend != "glfw":
            try:
                import mujoco.egl as egl
                from OpenGL import GL

                self.egl = egl
                self.gl = GL
                self.egl_context = egl.GLContext(width, height)
                self.egl_context.make_current()
                self.renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
                if any(name in self.renderer_name.lower() for name in ("llvmpipe", "softpipe", "software")):
                    raise RuntimeError("EGL selected a software renderer: " + self.renderer_name)
                self.release_egl()
            except Exception as error:
                if self.egl_context is not None:
                    self.egl_context.free()
                    self.egl_context = None
                if render_backend == "egl":
                    raise RuntimeError("GPU EGL initialization failed") from error
                self.backend_warning = str(error)
        self.render_backend = "egl" if self.egl_context is not None else "glfw"
        previous_vendor = os.environ.get("__GLX_VENDOR_LIBRARY_NAME")
        if self.egl_context is not None:
            os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "mesa"
        try:
            if not glfw.init():
                raise RuntimeError("GLFW initialization failed")
            glfw.window_hint(glfw.SAMPLES, 0 if self.egl_context is not None else 4)
            glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
            if hasattr(glfw, "SCALE_TO_MONITOR"):
                glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
            self.window = glfw.create_window(int(width), int(height), title, None, None)
        finally:
            if previous_vendor is None:
                os.environ.pop("__GLX_VENDOR_LIBRARY_NAME", None)
            else:
                os.environ["__GLX_VENDOR_LIBRARY_NAME"] = previous_vendor
        if not self.window:
            glfw.terminate()
            raise RuntimeError("MuJoCo keyboard window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        if self.egl_context is None:
            get_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_uint)(glfw.get_proc_address("glGetString"))
            self.renderer_name = get_string(0x1F01).decode()
        else:
            glfw.make_context_current(None)
            self.egl_context.make_current()
        self.cam = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        visual_meshes = {(int(model.geom_bodyid[index]), int(model.geom_dataid[index]),
                          tuple(model.geom_pos[index]), tuple(model.geom_quat[index]))
                         for index in range(model.ngeom)
                         if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH and
                         not model.geom_contype[index] and not model.geom_conaffinity[index]}
        self.hidden_geoms = {index for index in range(model.ngeom)
                             if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH and
                             (model.geom_contype[index] or model.geom_conaffinity[index]) and
                             (int(model.geom_bodyid[index]), int(model.geom_dataid[index]),
                              tuple(model.geom_pos[index]), tuple(model.geom_quat[index])) in visual_meshes}
        self.perturb = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        model.vis.quality.shadowsize = max(4096, model.vis.quality.shadowsize)
        model.vis.quality.offsamples = max(4, model.vis.quality.offsamples)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        if self.egl_context is not None:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        self.texts = []
        self.render_texts = []
        self.framebuffer_size = glfw.get_framebuffer_size(self.window)
        self.render_lock = threading.Lock()
        self.render_pending = threading.Event()
        self.render_stop = threading.Event()
        self.render_error = None
        self.render_thread = None
        self.latest_frame = None
        self.presented_frame = None
        self.presented_count = 0
        self.render_ms = 0.0
        self.present_ms = 0.0
        self.fps = 0.0
        self.last_present = None
        self.last_cursor = None
        self.last_click_time = -1.0
        self.last_click_position = None
        self.key_callback = key_callback
        glfw.set_key_callback(self.window, self.key_event)
        glfw.set_window_focus_callback(self.window, self.focus_changed)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_scroll_callback(self.window, lambda window, horizontal, vertical:
                                 mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                                       0, -0.05 * vertical, self.scene, self.cam))
        if asynchronous:
            if self.egl_context is not None:
                self.release_egl()
                glfw.make_context_current(self.window)
            else:
                glfw.make_context_current(None)
            self.render_thread = threading.Thread(target=self.render_loop, name="mujoco-render", daemon=True)
            self.render_thread.start()

    def release_egl(self):
        egl = self.egl
        egl.EGL.eglMakeCurrent(egl.EGL_DISPLAY, egl.EGL.EGL_NO_SURFACE,
                               egl.EGL.EGL_NO_SURFACE, egl.EGL.EGL_NO_CONTEXT)

    def key_event(self, window, key, scancode, action, mods):
        if hasattr(self.bridge, "set_keyboard_key"):
            self.bridge.set_keyboard_key(key, action != self.glfw.RELEASE)
        else:
            self.bridge.key_event(key, action != self.glfw.RELEASE)
        if action in (self.glfw.PRESS, self.glfw.RELEASE) and self.key_callback is not None:
            self.key_callback(key, "press" if action == self.glfw.PRESS else "release")

    def focus_changed(self, window, focused):
        if not focused:
            self.bridge.clear_keys()
            with self.state_lock:
                self.perturb.active = 0
                self.data.xfrc_applied[:] = 0
            self.last_cursor = None

    def select_body(self, horizontal, vertical):
        width, height = self.glfw.get_window_size(self.window)
        if width <= 0 or height <= 0:
            return
        point = np.zeros(3, dtype=np.float64)
        geom_id = np.full(1, -1, dtype=np.int32)
        flex_id = np.full(1, -1, dtype=np.int32)
        skin_id = np.full(1, -1, dtype=np.int32)
        with self.state_lock:
            body_id = mujoco.mjv_select(self.model, self.data, self.option, width / height,
                                       horizontal / width, 1 - vertical / height, self.scene,
                                       point, geom_id, flex_id, skin_id)
            self.perturb.active = 0
            if body_id > 0:
                self.perturb.select = body_id
                self.perturb.skinselect = int(skin_id[0])
                rotation = self.data.xmat[body_id].reshape(3, 3)
                self.perturb.localpos[:] = rotation.T @ (point - self.data.xpos[body_id])

    def mouse_button(self, window, button, action, mods):
        glfw = self.glfw
        if action == glfw.RELEASE:
            with self.state_lock:
                self.perturb.active = 0
                self.data.xfrc_applied[:] = 0
            return
        horizontal, vertical = glfw.get_cursor_pos(window)
        self.last_cursor = (horizontal, vertical)
        control = bool(mods & glfw.MOD_CONTROL)
        now = glfw.get_time()
        double_click = (button == glfw.MOUSE_BUTTON_LEFT and self.last_click_position is not None and
                        now - self.last_click_time < 0.3 and
                        np.hypot(horizontal - self.last_click_position[0],
                                 vertical - self.last_click_position[1]) < 5)
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.last_click_time = now
            self.last_click_position = (horizontal, vertical)
        if double_click or control and button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            self.select_body(horizontal, vertical)
        with self.state_lock:
            if control and self.perturb.select > 0:
                if button == glfw.MOUSE_BUTTON_LEFT:
                    self.perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
                elif button == glfw.MOUSE_BUTTON_RIGHT:
                    self.perturb.active = mujoco.mjtPertBit.mjPERT_ROTATE
                if self.perturb.active:
                    mujoco.mjv_initPerturb(self.model, self.data, self.scene, self.perturb)

    def apply_perturbation(self):
        with self.state_lock:
            self.data.xfrc_applied[:] = 0
            if self.perturb.active:
                mujoco.mjv_applyPerturbForce(self.model, self.data, self.perturb)

    def mouse_move(self, window, horizontal, vertical):
        previous = self.last_cursor
        self.last_cursor = (horizontal, vertical)
        if previous is None:
            return
        height = max(1, self.glfw.get_window_size(window)[1])
        if self.perturb.active:
            shifted = any(self.glfw.get_key(window, key) == self.glfw.PRESS
                          for key in (self.glfw.KEY_LEFT_SHIFT, self.glfw.KEY_RIGHT_SHIFT))
            if self.perturb.active == mujoco.mjtPertBit.mjPERT_TRANSLATE:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shifted else mujoco.mjtMouse.mjMOUSE_MOVE_V
            else:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shifted else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            with self.state_lock:
                mujoco.mjv_movePerturb(self.model, self.data, action,
                                      (horizontal - previous[0]) / height,
                                      (vertical - previous[1]) / height, self.scene, self.perturb)
            return
        if self.glfw.get_mouse_button(window, self.glfw.MOUSE_BUTTON_LEFT) != self.glfw.PRESS:
            return
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                             (horizontal - previous[0]) / height,
                             (vertical - previous[1]) / height, self.scene, self.cam)

    def poll_events(self):
        self.glfw.poll_events()

    def is_running(self):
        if self.render_error is not None:
            raise RuntimeError("MuJoCo rendering failed") from self.render_error
        return not self.glfw.window_should_close(self.window)

    def set_texts(self, texts):
        self.texts = texts

    def sync(self):
        self.glfw.poll_events()
        size = self.glfw.get_framebuffer_size(self.window)
        with self.render_lock, self.state_lock:
            self.framebuffer_size = size
            self.render_texts = list(self.texts)
            mujoco.mjv_updateScene(self.model, self.data, self.option, self.perturb, self.cam,
                                  mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            for geometry in self.scene.geoms[:self.scene.ngeom]:
                if geometry.objtype == mujoco.mjtObj.mjOBJ_GEOM and geometry.objid in self.hidden_geoms:
                    geometry.rgba[3] = 0
        if self.render_thread is not None:
            self.render_pending.set()
            if self.egl_context is not None:
                self.present_frame()
        else:
            if self.egl_context is not None:
                self.glfw.make_context_current(None)
                self.egl_context.make_current()
                self.capture_frame()
                self.release_egl()
                self.glfw.make_context_current(self.window)
                self.present_frame()
            else:
                self.glfw.make_context_current(self.window)
                self.draw_scene()
                self.glfw.swap_buffers(self.window)
                self.frame_presented()

    def draw_scene(self):
        width, height = self.framebuffer_size
        if width <= 0 or height <= 0:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        if self.egl_context is not None and (width != self.context.offWidth or height != self.context.offHeight):
            mujoco.mjr_resizeOffscreen(width, height, self.context)
        mujoco.mjr_render(viewport, self.scene, self.context)
        for font_scale, position, title, text in self.render_texts:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, position, viewport, title, text, self.context)

    def capture_frame(self):
        started = time.perf_counter()
        self.draw_scene()
        width, height = self.framebuffer_size
        if width <= 0 or height <= 0:
            return
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(pixels, None, mujoco.MjrRect(0, 0, width, height), self.context)
        self.render_ms = (time.perf_counter() - started) * 1000
        self.latest_frame = pixels

    def present_frame(self):
        pixels = self.latest_frame
        if pixels is None or pixels is self.presented_frame:
            return
        started = time.perf_counter()
        height, width = pixels.shape[:2]
        self.gl.glViewport(0, 0, width, height)
        self.gl.glPixelStorei(self.gl.GL_UNPACK_ALIGNMENT, 1)
        self.gl.glWindowPos2i(0, 0)
        self.gl.glDrawPixels(width, height, self.gl.GL_RGB, self.gl.GL_UNSIGNED_BYTE, pixels)
        self.glfw.swap_buffers(self.window)
        self.present_ms = (time.perf_counter() - started) * 1000
        self.presented_frame = pixels
        self.frame_presented()

    def frame_presented(self):
        now = time.perf_counter()
        if self.last_present is not None:
            self.fps = 1 / max(1e-9, now - self.last_present)
        self.last_present = now
        self.presented_count += 1

    def render_loop(self):
        try:
            if self.egl_context is not None:
                self.egl_context.make_current()
            else:
                self.glfw.make_context_current(self.window)
            while not self.render_stop.is_set():
                self.render_pending.wait()
                self.render_pending.clear()
                if self.render_stop.is_set():
                    break
                with self.render_lock:
                    if self.egl_context is not None:
                        self.capture_frame()
                    else:
                        self.draw_scene()
                if self.egl_context is None:
                    self.glfw.swap_buffers(self.window)
                    self.frame_presented()
        except Exception as error:
            self.render_error = error
        finally:
            if self.egl_context is not None:
                self.release_egl()
            else:
                self.glfw.make_context_current(None)

    def close(self):
        self.render_stop.set()
        self.render_pending.set()
        if self.render_thread is not None:
            self.render_thread.join()
        if self.egl_context is not None:
            self.glfw.make_context_current(None)
            self.egl_context.make_current()
        else:
            self.glfw.make_context_current(self.window)
        self.bridge.clear_keys()
        self.perturb.active = 0
        self.data.xfrc_applied[:] = 0
        self.context.free()
        if self.egl_context is not None:
            self.egl_context.free()
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
