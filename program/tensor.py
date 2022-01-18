import cv2
import os
import numpy as np
from PIL import Image
import time

from detector import DetectorTF2

# For preprocessing image
def make_image_square(filename):
    img = cv2.imread(filename)
    # Size of the image
    # s = max(img.shape[0:2])
    s = 640

    # Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)

    # Getting the centering position
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    # Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
    cv2.imwrite(filename, f)

# For preprocessing image
def crop_image():
    for image_index in range(2):
        folder_name = 'data/'
        image_name = folder_name + str(image_index + 1) + '.jpg'
        img = cv2.imread(image_name)

        h, w, c = img.shape

        w_constant = w/3
        h_constant = h/2

        image_part_index = 0

        for index_w in range(3):
            for index_h in range(2):
                start_width = int(w_constant * index_w)
                end_width = int(w_constant * (index_w + 1))

                start_height = int(h_constant * index_h)
                end_height = int(h_constant * (index_h + 1))

                current_index = image_part_index

                # For training image set
                # section_name = 'PL_8_' + str(image_index+1) + '_'
                # file_name = section_name + str(image_index+1) + '_' + str(image_part_index) + '.jpg'

                # For testing image set
                section_name = str(image_index+1) + '/'
                file_name = folder_name + section_name + \
                    str(image_part_index+1) + '.jpg'

                crop_img = img[start_height:end_height, start_width:end_width]

                image_part_index = image_part_index + 1
                cv2.imwrite(file_name, crop_img)

                make_image_square(file_name)

# For detection
def WriteFile(output_dir, file_name, content):
    file_output = os.path.join(output_dir, file_name)
    f = open(file_output, 'a+')
    f.write(content)
    f.close()

# For detection
def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/'):
	total_detected = 0
	timestamp2 = time.time()

	for file in os.scandir(images_dir):
		if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')) :
			image_path = os.path.join(images_dir, file.name)
			img = cv2.imread(image_path)
			timestamp1 = time.time()
			det_boxes = detector.DetectFromImage(img)
			elapsed_time = round((time.time() - timestamp1) * 1000) #ms
			img = detector.DisplayDetections(img, det_boxes)

			total_detected = total_detected + len(det_boxes)
			text_to_save = str(file.name) + ':\t' + str(len(det_boxes)) + ' benur detected' + '\t' + '[' + str(elapsed_time/1000) + ' s] \t\n'

			if save_output:
				img_out = os.path.join(output_dir, file.name)
				cv2.imwrite(img_out, img)
				WriteFile(output_dir, 'ResultLog.txt', text_to_save)

	elapsed_time2 = round((time.time() - timestamp2) * 1000) #ms
	final_text_to_save = str(total_detected) + 'benur detected\t' + '[' + str(elapsed_time2/1000) + ' s]'
	if save_output:
		WriteFile(output_dir, 'Final.txt', final_text_to_save)
	return total_detected

# For detection
def execute_tf(model_path, threshold, output_directory, labelmap_path, images_dir, id_list_data = None):
    id_list = id_list_data
    if id_list_data is not None:
        id_list = [int(item) for item in id_list_data.split(',')]

    save_output = True
    if save_output:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # instance of the class DetectorTF2
    detector = DetectorTF2(model_path, labelmap_path,
                            class_id=id_list, threshold=threshold)

    result = DetectImagesFromFolder(
        detector, images_dir, save_output=True, output_dir=output_directory)

    return result


models = ['ssd-mobilenet-v2-5000']
threshold_setup = [0.3]
test_images_folders = ['1', '2']

# For detection
def detect_images():
    detected_total = 0
    total_detection_process = 0
    for threshold in threshold_setup:
        # Generate string for threshold output folder
        threshold_str = str(threshold)
        threshold_str = threshold_str.replace('.', '_')

        for folder in test_images_folders:

            # Generate string for output folder
            folder_subname = folder.replace('/', '_')

            for model in models:
                # Generate output directory
                output_directory = 'output_' + folder_subname + '_' + threshold_str

                detection_model_path = 'models/' + model
                detection_labelmap_path = 'models/shrimp-seed_label_map.pbtxt'
                detection_images_dir = 'data/' + folder
                detection_output_dir = 'data/' + output_directory + '/' + model

                detection_result = execute_tf(detection_model_path, threshold, detection_output_dir, detection_labelmap_path, detection_images_dir)

                detected_total += int(detection_result)
                total_detection_process += 1

    return float(detected_total) / float(total_detection_process)


def main():

    #=========== Detection Program here ========#
    crop_image()
    #=========== Should return the result ======#
    detected_result = detect_images()

    print(detected_result, end='')


if __name__ == "__main__":
    main()
