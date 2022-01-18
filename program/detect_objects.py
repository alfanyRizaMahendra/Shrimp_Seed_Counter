import os
import cv2
import time
import argparse

from detector import DetectorTF2

def WriteFile(output_dir, file_name, content):
	file_output = os.path.join(output_dir, file_name)
	f = open(file_output, 'a+')
	f.write(content)
	f.close()


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
	print(total_detected)
	if save_output:
		WriteFile(output_dir, 'Final.txt', final_text_to_save)



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
	parser.add_argument('--model_path', help='Path to frozen detection model',
						default='models/efficientdet_d0_coco17_tpu-32/saved_model')
	parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file',
	                    default='models/mscoco_label_map.pbtxt')
	parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","',
	                    type=str, default=None) # example input "1,3" to detect person and car
	parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.4)
	parser.add_argument('--images_dir', help='Directory to input images)', default='data/samples/images/')
	parser.add_argument('--output_directory', help='Path to output images and video', default='data/samples/output')
	parser.add_argument('--save_output', help='Flag for save images and video with detections visualized, default: False',
	                    action='store_true')  # default is false
	args = parser.parse_args()

	id_list = None
	if args.class_ids is not None:
		id_list = [int(item) for item in args.class_ids.split(',')]

	if args.save_output:
		if not os.path.exists(args.output_directory):
			os.makedirs(args.output_directory)

	# instance of the class DetectorTF2
	detector = DetectorTF2(args.model_path, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)

	DetectImagesFromFolder(detector, args.images_dir, save_output=args.save_output, output_dir=args.output_directory)