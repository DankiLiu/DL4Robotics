import classic_framework.mujoco.mujoco_utils.mujoco_controllers as mj_ctrl
from classic_framework.mujoco.MujocoRobot import MujocoRobot
from classic_framework.mujoco.MujocoScene import MujocoScene as Scene
from classic_framework.mujoco.mujoco_utils.mujoco_scene_object import MujocoPrimitiveObject

if __name__ == '__main__':
    box1 = MujocoPrimitiveObject(obj_pos=[.5, -0.2, 0.35], obj_name="box1", geom_rgba=[0.1, 0.25, 0.3, 1])
    box2 = MujocoPrimitiveObject(obj_pos=[.6, -0.1, 0.35], obj_name="box2", geom_rgba=[0.2, 0.3, 0.7, 1])
    box3 = MujocoPrimitiveObject(obj_pos=[.4, -0.1, 0.35], obj_name="box3", geom_rgba=[1, 0, 0, 1])
    box4 = MujocoPrimitiveObject(obj_pos=[.6, -0.0, 0.35], obj_name="box4", geom_rgba=[1, 0, 0, 1])
    box5 = MujocoPrimitiveObject(obj_pos=[.6, 0.1, 0.35], obj_name="box5", geom_rgba=[1, 1, 1, 1])
    box6 = MujocoPrimitiveObject(obj_pos=[.6, 0.2, 0.35], obj_name="box6", geom_rgba=[1, 0, 0, 1])

    table = MujocoPrimitiveObject(obj_pos=[0.5, 0.0, 0.2],
                                  obj_name="table0",
                                  geom_size=[0.25, 0.35, 0.2],
                                  mass=2000)

    object_list = [box1, box2, box3, box4, box5, box6, table]

    scene = Scene(object_list=object_list, control=mj_ctrl.MocapControl())  # if we want to do mocap control
    # scene = Scene(object_list=object_list)                        # ik control is default

    duration = 1
    mj_Robot = MujocoRobot(scene, gravity_comp=True, num_DoF=7)

    mj_Robot.set_gripper_width = 0.0
    init_pos = mj_Robot.current_c_pos
    init_or = mj_Robot.current_c_quat

    mj_Robot.gotoCartPositionAndQuat([0.5, -0.2, 0.6 - 0.1], [0, 1, 0, 0], duration=duration)
    mj_Robot.set_gripper_width = 0.04
    mj_Robot.gotoCartPositionAndQuat([0.5, -0.2, 0.52 - 0.1], [0, 1, 0, 0], duration=duration)
    mj_Robot.set_gripper_width = 0.00
    mj_Robot.gotoCartPositionAndQuat(init_pos, init_or, duration=duration)

    # Inhand Camera Point Clouds
    points_inhand, colors_inhand = scene.get_point_cloud_inHandCam()
    scene.visualize_point_clouds(points_inhand, colors_inhand)

    mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.6 - 0.1], [0, 1, 0, 0], duration=duration)
    mj_Robot.set_gripper_width = 0.04

    mj_Robot.gotoCartPositionAndQuat([0.6, -0.1, 0.6 - 0.1], [0, 1, 0, 0], duration=duration)
    mj_Robot.gotoCartPositionAndQuat([0.6, -0.1, 0.52 - 0.1], [0, 1, 0, 0], duration=duration)
    mj_Robot.set_gripper_width = 0.00

    mj_Robot.gotoCartPositionAndQuat(init_pos, init_or, duration=duration)
    mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.6 - 0.1], [0, 1, 0, 0], duration=duration)
    mj_Robot.set_gripper_width = 0.04

    mj_Robot.gotoCartPositionAndQuat([.4, -0.1, 0.6 - 0.1], [0, 1, 0, 0], duration=duration)
    mj_Robot.gotoCartPositionAndQuat([.4, -0.1, 0.52 - 0.1], [0, 1, 0, 0], duration=duration)
    mj_Robot.set_gripper_width = 0.00

    mj_Robot.gotoCartPositionAndQuat(init_pos, init_or, duration=duration)

    # Cage Camera Point Clouds
    points_cage, colors_cage = scene.get_point_cloud_CageCam()
    scene.visualize_point_clouds(points_cage, colors_cage)

    mj_Robot.gotoCartPositionAndQuat([0.5, 0.2, 0.65 - 0.1], [0, 1, 0, 0], duration=duration)
    mj_Robot.set_gripper_width = 0.04

    mj_Robot.gotoCartPositionAndQuat(init_pos, init_or, duration=duration)
