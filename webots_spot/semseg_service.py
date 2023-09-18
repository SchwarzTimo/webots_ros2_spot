import rclpy
from rclpy.node import Node
import tensorflow as tf
import ros2_numpy as rnp
import numpy as np

from custom_msgs.srv import ExecuteSemSeg
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int64

#from ament_index_python.packages import get_package_prefix


#Fixed=?
raw_image_topic = 'Spot/kinect_color/image_color' #later from .urdf or similiar

#package_path = get_package_prefix('webots_spot')
package_path = '/home/timo/ws/webots_spot/src/webots_spot/models/' #will be changed to auto find


class SemSegService(Node):

    def __init__(self):
        super().__init__('semseg_service')
        self.srv = self.create_service(ExecuteSemSeg, 'execute_semseg', self.execute_semseg_callback)
        self.subscription = self.create_subscription(Image, raw_image_topic, self.image_callback, 1) 
        self.publisher = 'init'
        self.declare_parameter('model_name', str())
        self.declare_parameter('publish_topic_name', str())
        #set import path and load tf model
        import_path = package_path + self.get_parameter('model_name').value
        self.model = tf.keras.models.load_model(import_path)   
        #self.get_logger().info(str())
          

    def execute_semseg_callback(self, request, response):
        #getting prediction
        prediction = self.get_prediction()
        
        #create publisher with given name and type
        #and change tensor to corresponding msg type
        if request.only_most_likely:
            if self.publisher == 'init':
                self.publisher = self.create_publisher(Image, self.get_parameter('publish_topic_name').value, 1)
            pred = self.create_mask(prediction).numpy()
            pred = pred.astype(np.int16)
            msg = rnp.msgify(Image, pred, '16SC1')
        else:
            self.publisher = self.create_publisher(Float64MultiArray, self.get_parameter('model_name').value, 1)
            msg = Float64MultiArray()
            msg.data = prediction
 
        
        #publish prediction to topic
        msg.header.frame_id = "kinect_color"
        self.publisher.publish(msg)

        #respond service call with the name of the prediction topic
        response.used_model_name = self.get_parameter('model_name').value
        response.published_topic_name = self.get_parameter('publish_topic_name').value  
        return response
    
    def image_callback(self, msg):
        self.last_image = msg

    def get_prediction(self):      
        tensor = self.get_tensor()
        #shape_of_input_tensor_of_model_test2 = (1, 224, 224, 3)
        pred_mask = self.model.predict((tensor[tf.newaxis, ...]))
        #shape of output tensor of model_test2 = (1, 224, 224, 3)
        #first dimension not needed anymore
        pred_mask = tf.squeeze(pred_mask)
        #shape (imgX, imgY, num_classes); case model_test2: (224, 224, 3)
        return pred_mask
    
    def get_tensor(self):
        #ros2 msg/Image to numpy array
        img = rnp.numpify(self.last_image)
        #(x, y, 4) -> (x, y, 3)
        img = img[:,:,0:3]
        #resize image to fit input requirement of model
        IMG_SIZE=(224, 224)
        img = tf.image.resize(img, size=IMG_SIZE)
        #normalize
        img = tf.cast(img, dtype=tf.float32) / 255.0

        return img
    
    def create_mask(self, pred_mask):
        #get most likely class
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask

def main():
    rclpy.init()
    semseg_service = SemSegService()
    
    rclpy.spin(semseg_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
