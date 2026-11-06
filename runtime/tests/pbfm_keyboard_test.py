from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/sim"))
from pbfm_smoke import KeyboardBridge, KeyboardViewer
from mujoco_keyboard_viewer import KeyboardViewer as SharedKeyboardViewer


class MousePerturbationTest(unittest.TestCase):
    def setUp(self):
        import glfw

        self.viewer = SharedKeyboardViewer.__new__(SharedKeyboardViewer)
        self.viewer.model = mujoco.MjModel.from_xml_string('''
            <mujoco><worldbody><body name="robot" pos="0 0 0.8">
              <freejoint/><geom type="sphere" size="0.2" mass="1"/>
            </body></worldbody></mujoco>''')
        self.viewer.data = mujoco.MjData(self.viewer.model)
        mujoco.mj_forward(self.viewer.model, self.viewer.data)
        self.viewer.cam = mujoco.MjvCamera()
        self.viewer.cam.lookat[:] = [0, 0, 0.8]
        self.viewer.cam.distance = 3
        self.viewer.option = mujoco.MjvOption()
        self.viewer.perturb = mujoco.MjvPerturb()
        self.viewer.scene = mujoco.MjvScene(self.viewer.model, maxgeom=100)
        mujoco.mjv_updateScene(self.viewer.model, self.viewer.data, self.viewer.option,
                              self.viewer.perturb, self.viewer.cam, mujoco.mjtCatBit.mjCAT_ALL,
                              self.viewer.scene)
        self.viewer.glfw = Mock(wraps=glfw)
        for name in ("RELEASE", "PRESS", "MOD_CONTROL", "MOUSE_BUTTON_LEFT", "MOUSE_BUTTON_RIGHT",
                     "KEY_LEFT_SHIFT", "KEY_RIGHT_SHIFT"):
            setattr(self.viewer.glfw, name, getattr(glfw, name))
        self.viewer.glfw.get_window_size.return_value = (800, 600)
        self.viewer.glfw.get_cursor_pos.return_value = (400, 300)
        self.viewer.glfw.get_time.return_value = 1.0
        self.viewer.glfw.get_key.return_value = glfw.RELEASE
        self.viewer.window = None
        self.viewer.bridge = Mock()
        self.viewer.state_lock = threading.RLock()
        self.viewer.last_cursor = None
        self.viewer.last_click_time = -1.0
        self.viewer.last_click_position = None

    def test_ctrl_drag_applies_force_and_release_clears_it(self):
        glfw = self.viewer.glfw
        self.viewer.mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, glfw.MOD_CONTROL)
        self.assertEqual(self.viewer.perturb.select, 1)
        self.assertEqual(self.viewer.perturb.active, mujoco.mjtPertBit.mjPERT_TRANSLATE)
        initial_position = self.viewer.data.qpos.copy()
        self.viewer.mouse_move(None, 450, 320)
        self.viewer.apply_perturbation()
        self.assertGreater(np.linalg.norm(self.viewer.data.xfrc_applied), 0)
        np.testing.assert_array_equal(self.viewer.data.qpos, initial_position)
        mujoco.mj_step(self.viewer.model, self.viewer.data)
        self.assertFalse(np.array_equal(self.viewer.data.qpos[:2], initial_position[:2]))
        self.viewer.mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
        self.assertEqual(self.viewer.perturb.active, 0)
        self.assertFalse(self.viewer.data.xfrc_applied.any())

    def test_ctrl_rotate_and_focus_loss(self):
        glfw = self.viewer.glfw
        self.viewer.mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, glfw.MOD_CONTROL)
        self.assertEqual(self.viewer.perturb.active, mujoco.mjtPertBit.mjPERT_ROTATE)
        self.viewer.mouse_move(None, 450, 320)
        self.viewer.apply_perturbation()
        self.assertGreater(np.linalg.norm(self.viewer.data.xfrc_applied[:, 3:]), 0)
        self.viewer.focus_changed(None, False)
        self.viewer.bridge.clear_keys.assert_called_once()
        self.assertEqual(self.viewer.perturb.active, 0)
        self.assertFalse(self.viewer.data.xfrc_applied.any())

    def test_double_click_selects_without_force(self):
        glfw = self.viewer.glfw
        self.viewer.mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
        self.viewer.mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
        glfw.get_time.return_value = 1.2
        self.viewer.mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
        self.assertEqual(self.viewer.perturb.select, 1)
        self.assertEqual(self.viewer.perturb.active, 0)


class KeyboardTest(unittest.TestCase):
    def setUp(self):
        self.bridge = KeyboardBridge.__new__(KeyboardBridge)
        self.bridge.keyboard_lock = threading.Lock()
        self.bridge.held_keys = set()
        self.bridge.speed = self.bridge.lateral = self.bridge.turn = 0.0
        self.bridge.start_requested = self.bridge.exit_requested = False

    def test_hold_repeat_and_release(self):
        for repeat in range(10):
            self.bridge.key_event(ord("W"), True)
            self.assertEqual(self.bridge.speed, 1.0)
        self.bridge.key_event(ord("W"), False)
        np.testing.assert_array_equal(self.bridge._joystick_mask_and_axes()[1], np.zeros(4))

    def test_release_one_axis_keeps_other_axis(self):
        self.bridge.key_event(ord("W"), True)
        self.bridge.key_event(ord("A"), True)
        self.bridge.key_event(ord("Q"), True)
        np.testing.assert_allclose(self.bridge._joystick_mask_and_axes()[1], [-1, -1, 0, 1])
        self.bridge.key_event(ord("A"), False)
        np.testing.assert_allclose(self.bridge._joystick_mask_and_axes()[1], [-1, 0, 0, 1])
        self.bridge.key_event(ord("W"), False)
        self.bridge.key_event(ord("Q"), False)
        np.testing.assert_array_equal(self.bridge._joystick_mask_and_axes()[1], np.zeros(4))

    def test_translation_does_not_set_turn(self):
        for key in "WSQE":
            self.bridge.clear_keys()
            self.bridge.key_event(ord(key), True)
            self.assertEqual(self.bridge._joystick_mask_and_axes()[1][1], 0.0)

    def test_focus_loss_clears_all_axes(self):
        for key in "WAE":
            self.bridge.key_event(ord(key), True)
        self.bridge.clear_keys()
        self.assertFalse(self.bridge.held_keys)
        np.testing.assert_array_equal(self.bridge._joystick_mask_and_axes()[1], np.zeros(4))

    def test_opposite_keys_and_space(self):
        for key in "WSADQE":
            self.bridge.key_event(ord(key), True)
        np.testing.assert_array_equal(self.bridge._joystick_mask_and_axes()[1], np.zeros(4))
        self.bridge.key_event(ord("W"), False)
        self.assertEqual(self.bridge.speed, -0.3)
        self.bridge.key_event(32, True)
        np.testing.assert_array_equal(self.bridge._joystick_mask_and_axes()[1], np.zeros(4))

    def test_start_and_exit(self):
        self.bridge.key_event(257, True)
        self.bridge.key_event(257, False)
        self.assertTrue(self.bridge.start_requested)
        self.bridge.key_event(ord("X"), True)
        self.assertTrue(self.bridge.exit_requested)

    def test_shared_viewer_keyboard_entry_updates_pbfm_bridge(self):
        self.bridge.set_keyboard_key(ord("W"), True)
        self.assertEqual(self.bridge.speed, 1.0)
        self.bridge.set_keyboard_key(257, True)
        self.bridge.set_keyboard_key(ord("X"), True)
        self.assertTrue(self.bridge.start_requested)
        self.assertTrue(self.bridge.exit_requested)

    def test_analog_axes_and_speed_limit(self):
        for amplitude in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.bridge.set_axes(amplitude, 0, 0)
            self.assertEqual(self.bridge.speed, amplitude)
            np.testing.assert_allclose(self.bridge._joystick_mask_and_axes()[1], [0, 0, 0, amplitude])
        self.bridge.set_axes(2, 0, -0.25)
        self.assertEqual(self.bridge.speed, 1.0)
        np.testing.assert_allclose(self.bridge._joystick_mask_and_axes()[1], [0, 0.25, 0, 1])


class RenderSchedulingTest(unittest.TestCase):
    def test_slow_buffer_swap_does_not_lock_scene_updates(self):
        viewer = SharedKeyboardViewer.__new__(SharedKeyboardViewer)
        viewer.glfw = Mock()
        viewer.window = object()
        viewer.egl_context = None
        viewer.frame_presented = Mock()
        viewer.render_lock = threading.Lock()
        viewer.render_pending = threading.Event()
        viewer.render_stop = threading.Event()
        viewer.render_error = None
        viewer.draw_scene = Mock()
        swapping = threading.Event()
        finish_swap = threading.Event()

        def swap_buffers(window):
            swapping.set()
            finish_swap.wait(timeout=2)

        viewer.glfw.swap_buffers.side_effect = swap_buffers
        renderer = threading.Thread(target=viewer.render_loop)
        renderer.start()
        try:
            viewer.render_pending.set()
            self.assertTrue(swapping.wait(timeout=1))
            acquired = viewer.render_lock.acquire(timeout=0.1)
            self.assertTrue(acquired, "display presentation blocked physics scene updates")
            if acquired:
                viewer.render_lock.release()
        finally:
            viewer.render_stop.set()
            viewer.render_pending.set()
            finish_swap.set()
            renderer.join(timeout=2)
        self.assertFalse(renderer.is_alive())
        self.assertIsNone(viewer.render_error)
        viewer.glfw.make_context_current.assert_any_call(None)

    def test_render_failures_reach_the_simulation_loop(self):
        viewer = SharedKeyboardViewer.__new__(SharedKeyboardViewer)
        viewer.glfw = Mock()
        viewer.window = object()
        viewer.egl_context = None
        viewer.render_lock = threading.Lock()
        viewer.render_pending = threading.Event()
        viewer.render_stop = threading.Event()
        viewer.render_error = None
        viewer.draw_scene = Mock(side_effect=RuntimeError("render failed"))
        viewer.render_pending.set()
        renderer = threading.Thread(target=viewer.render_loop)
        renderer.start()
        renderer.join(timeout=1)
        self.assertFalse(renderer.is_alive())
        with self.assertRaisesRegex(RuntimeError, "MuJoCo rendering failed"):
            viewer.is_running()
        viewer.glfw.swap_buffers.assert_not_called()


if __name__ == "__main__":
    unittest.main()
