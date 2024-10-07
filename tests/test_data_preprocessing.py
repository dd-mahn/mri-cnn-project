import unittest
from data_preprocessing import load_and_preprocess_image

class TestDataPreprocessing(unittest.TestCase):
    def test_image_loading(self):
        path = 'path_to_test_image.jpg'
        image = load_and_preprocess_image(path)
        self.assertEqual(image.shape, (224, 224))

if __name__ == '__main__':
    unittest.main()