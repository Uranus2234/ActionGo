from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/sim"))
from pbfm_smoke import KeyboardBridge, KeyboardViewer


class MousePerturbationTest(unittest.TestCase):
    def setUp(self):
        import glfw

        self.viewer = KeyboardViewer.__new__(KeyboardViewer)
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
        self.viewer.last_cursor = None
        self.viewer.last_click_time = -1.0
        self.viewer.last_click_position = None

    def test_ctrl_drag_applies_force_and_release_clears_it(self):
        glfw = self.viewer.glfw
        self.viewer.mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, glfw.MOD_CONTROL)
        self.assertEqual(self.viewer.perturb.select, 1)
        self.assertEqual(self.viewer.perturb.active, mujoco.mjtPertBit.mjPERT_TRANSLATE)
        initial_position = self.viewer.data.qpos.copy()
        self.viewer.mouse_move(None, 450, 320)
        self.viewer.apply_perturbation()
        self.assertGreater(np.linalg.norm(self.viewer.data.xfrc_applied), 0)
        np.testing.assert_array_equal(self.viewer.data.qpos, initial_position)
        self.viewer.mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.RELEASE, 0)
        self.assertEqual(self.viewer.perturb.active, 0)
        self.assertFalse(self.viewer.data.xfrc_applied.any())

    def test_ctrl_rotate_and_focus_loss(self):
        glfw = self.viewer.glfw
        self.viewer.mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, glfw.MOD_CONTROL)
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

    def test_analog_axes_and_speed_limit(self):
        for amplitude in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.bridge.set_axes(amplitude, 0, 0)
            self.assertEqual(self.bridge.speed, amplitude)
            np.testing.assert_allclose(self.bridge._joystick_mask_and_axes()[1], [0, 0, 0, amplitude])
        self.bridge.set_axes(2, 0, -0.25)
        self.assertEqual(self.bridge.speed, 1.0)
        np.testing.assert_allclose(self.bridge._joystick_mask_and_axes()[1], [0, 0.25, 0, 1])


if __name__ == "__main__":
    unittest.main()
