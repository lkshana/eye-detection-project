import os
import unittest
from app import app
from PIL import Image

class EyeDiseaseAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
        # Create a dummy image for testing
        self.test_img_path = 'test_eye.png'
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(self.test_img_path)

    def tearDown(self):
        # Clean up
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)
            
    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'VisionAI', response.data)

    def test_analyze_route(self):
        with open(self.test_img_path, 'rb') as img:
            data = {
                'file': (img, 'test_eye.png')
            }
            response = self.app.post('/analyze', data=data, content_type='multipart/form-data', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Analysis Complete', response.data)
            # Check if PDF link is present
            self.assertIn(b'Download Full Report', response.data)

if __name__ == '__main__':
    unittest.main()
