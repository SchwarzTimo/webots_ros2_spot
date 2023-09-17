import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution


PACKAGE_NAME = 'webots_spot'

def generate_launch_description():

    model_name_launch_arg = DeclareLaunchArgument(
        'model_name', default_value=TextSubstitution(text='model_test2')
    )

    publish_topic_name_launch_arg = DeclareLaunchArgument(
        'publish_topic_name', default_value=TextSubstitution(text='Spot/prediction_mask')
    )

    semseg_service = Node(
        package=PACKAGE_NAME,
        executable='semseg_service',
        name='semseg_service',
        parameters=[{
            'model_name': LaunchConfiguration('model_name'),
            'publish_topic_name': LaunchConfiguration('publish_topic_name'),
        }]
    )

    return LaunchDescription([
        model_name_launch_arg,
        publish_topic_name_launch_arg,
        semseg_service,
    ])