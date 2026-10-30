import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def main():
    doc = Document()

    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)

    # Title
    heading = doc.add_heading('Mobile Robot Design and Simulation Lab Assignment', 0)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('1. Introduction', level=1)
    doc.add_paragraph("This report details the design, modeling, simulation, and control of a custom mobile robot using ROS (Robot Operating System). The project involves URDF/Xacro modeling, Gazebo simulation, RViz visualization, and the development of ROS nodes for automated and manual motion control.")

    doc.add_heading('2. Robot Modeling (URDF/Xacro)', level=1)
    doc.add_paragraph("The mobile robot is modeled using URDF/Xacro by defining physical properties and joints. The model comprises a base_link, caster_link, laser_link, and left/right wheel links. The structure was successfully validated using the check_urdf tool as shown below.")
    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-39-15.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 1: Successful parsing of the robot URDF showing the root link and its four child links.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('3. Simulation Environment Setup (Gazebo & RViz)', level=1)
    doc.add_paragraph("A custom simulation environment with boundary walls was designed to test the robot's sensors and motion within a confined space. The Gazebo world is defined using an SDF file, specifying the size and pose of the bounding walls.")
    
    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-48-28.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 2: SDF definitions for the simulation environment (Walls 1 & 2).")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-50-00.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 3: SDF definitions for the simulation environment (Wall 4).")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("To initiate the simulation, a ROS launch file was created to load the robot description, start Gazebo, spawn the robot model, and initialize RViz.")
    
    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-40-07.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 4: The rviz.launch file.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-50-05.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 5: The main launch file starting Gazebo and spawning the URDF model.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('3.1 Gazebo and RViz Initialization', level=2)
    doc.add_paragraph("Gazebo was launched with the defined world, and the robot spawned at the origin. RViz was configured to visualize the RobotModel and TF frames.")
    
    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-50-42.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 6: Gazebo simulation environment with walls and spawned robot.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-46-21.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 7: RViz configured with the RobotModel and Grid.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('4. Robot Motion Control and Testing', level=1)
    doc.add_paragraph("Initial testing of the robot's motion was performed by publishing velocity commands directly to the /cmd_vel topic via the terminal.")

    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-51-37.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 8: Testing forward motion via rostopic pub.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('4.1 Automated Motion Node', level=2)
    doc.add_paragraph("A Python script (robot_controller.py) was developed to automate basic operations such as forward, reverse, left, and right movements.")
    
    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 22-01-56.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 9: Implementation of the automated motion controller node.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 22-03-04.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 10: Execution output of the automated motion sequence.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('4.2 Teleoperation Node', level=2)
    doc.add_paragraph("To allow interactive manual control, a teleop node (mobile_robot_controller.py) was developed. It maps keyboard inputs (W, A, S, D, X) to corresponding linear and angular velocities.")
    
    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 22-26-30.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 11: Implementation of the teleoperation node.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 22-29-14.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 12: Robot being controlled via teleop, with RViz displaying odometry path tracking.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 22-30-54.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 13: Close-up of the teleoperation terminal output.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('5. Sensor and TF Analysis', level=1)
    doc.add_paragraph("The robot is equipped with a simulated LiDAR sensor and differential drive odometry. The relationship between the commanded velocity, actual robot motion, and sensor/TF data was analyzed.")

    doc.add_heading('5.1 LaserScan Visualization', level=2)
    doc.add_paragraph("The LiDAR rays accurately detect the boundary walls in Gazebo, which corresponds to the LaserScan data visualized in RViz, confirming proper sensor mounting and TF broadcasting.")

    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-57-28.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 14: Gazebo showing LiDAR rays interacting with the environment.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 22-07-34.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 15: RViz successfully rendering LaserScan data matching the Gazebo walls.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('5.2 Odometry and Pose Verification', level=2)
    doc.add_paragraph("By monitoring the /odom topic and querying Gazebo's internal state, the robot's pose and twist values were verified to align with the commanded velocities.")
    
    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-54-49.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 16: Odometry output reflecting twist variations.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 21-58-28.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 17: Querying Gazebo for the exact model pose and twist states.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_picture(r"c:\Users\G15\Downloads\attachments\Screenshot From 2026-10-08 22-19-58.png", width=Inches(6.0))
    p = doc.add_paragraph("Figure 18: RViz mapping Odometry topics while corresponding data is echoed in the terminal.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('6. Conclusion', level=1)
    doc.add_paragraph("The mobile robot system was successfully modeled, simulated, and controlled. The integration between Gazebo's physics simulation, RViz visualization, and the custom ROS nodes proved to be robust. Commanded velocities were accurately reflected in the robot's motion, while LiDAR and odometry sensors provided correct environmental and localization data, satisfying all assignment requirements.")

    doc.save(r'c:\Users\G15\Downloads\AI_project\Mobile_Robot_Lab_Assignment.docx')
    print("Document successfully created!")

if __name__ == '__main__':
    main()
