import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

xml_path = 'differentialdrive.xml'  # xml file (assumes this is in the same folder as this file)
simend = 10  # simulation time
print_camera_config = 0  # set to 1 to print camera config

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

desired_position = 5.0  # Desired position to reach
Kp = 1.0  # Proportional gain
Ki = 0.1  # Integral gain
Kd = 0.01  # Derivative gain

integral = 0.0
prev_error = 0.0


def quat2euler(quat_mujoco):
    # MuJoCo quat is constant, x, y, z,
    # scipy quat is x, y, z, constant
    quat_scipy = np.array([quat_mujoco[3], quat_mujoco[0], quat_mujoco[1], quat_mujoco[2]])
    r = R.from_quat(quat_scipy)
    euler = r.as_euler('xyz', degrees=True)
    return euler


def init_controller(model, data):
    global integral, prev_error
    integral = 0.0
    prev_error = 0.0


def pid_controller(current_position, dt):
    global integral, prev_error

    error = desired_position - current_position
    integral += error * dt
    derivative = (error - prev_error) / dt
    control_signal = Kp * error + Ki * integral + Kd * derivative
    prev_error = error

    return control_signal


def controller(model, data):
    current_position = data.qpos[0]  # Assuming qpos[0] is the position of interest
    dt = model.opt.timestep
    control_signal = pid_controller(current_position, dt)

    # Apply the control signal to both wheels to move forward/backward
    data.ctrl[0] = control_signal
    data.ctrl[1] = control_signal


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)


def mouse_button(window, button, act, mods):
    global button_left
    global button_middle
    global button_right
    global lastx
    global lasty

    if act == glfw.PRESS:
        button_left = (button == glfw.MOUSE_BUTTON_LEFT)
        button_middle = (button == glfw.MOUSE_BUTTON_MIDDLE)
        button_right = (button == glfw.MOUSE_BUTTON_RIGHT)
        lastx, lasty = glfw.get_cursor_pos(window)
    else:
        button_left = False
        button_middle = False
        button_right = False


def mouse_move(window, xpos, ypos):
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if (not button_left) and (not button_middle) and (not button_right):
        return

    width, height = glfw.get_window_size(window)

    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)


# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data
cam = mj.MjvCamera()  # Abstract camera
opt = mj.MjvOption()  # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = 90
cam.elevation = -45
cam.distance = 13
cam.lookat = np.array([0.0, 0.0, 0.0])

# initialize the controller
init_controller(model, data)

# set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while data.time - time_prev < 1.0 / 60.0:
        mj.mj_step(model, data)

    quat = np.array([data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]])
    euler = quat2euler(quat)

    print(data.site_xpos[0])

    if data.time >= simend:
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # print camera configuration (help to initialize the view)
    if print_camera_config == 1:
        print('cam.azimuth =', cam.azimuth, ';', 'cam.elevation =', cam.elevation, ';', 'cam.distance = ', cam.distance)
        print('cam.lookat =np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
