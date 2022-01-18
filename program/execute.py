import subprocess

models = [
    # 'faster-rcnn-resnet50-6000',
    # 'faster-rcnn-resnet50-5000',
    # 'faster-rcnn-resnet50-4000',
    # 'ssd-mobilenet-v2-4000',
    # 'ssd-mobilenet-v2-6000',
    'ssd-mobilenet-v2-5000',
]

threshold_setup = [0.3]
test_images_folders = ['1', '2']

for threshold in threshold_setup:
    
    # Generate string for threshold output folder
    threshold_str = str(threshold)
    threshold_str = threshold_str.replace('.', '_')

    for folder in test_images_folders:

        # Generate string for output folder
        folder_subname = folder.replace('/', '_')

        for model in models:
            print('start executing [folder: ' + folder + '] ' + ' [threshold: ' + str(threshold) + ']: ' + model )
            
            # Generate output directory
            output_directory = 'output_' + folder_subname + '_' + threshold_str

            # Generate command to execute [on terminal]
            commmand_to_execute = 'python3 detect_objects.py --threshold ' + str(threshold) +' --model_path models/' + model + ' --path_to_labelmap models/shrimp-seed_label_map.pbtxt --images_dir data/' + folder +' --output_directory data/' + output_directory + '/' + model + ' --save_output'
            subprocess_result = subprocess.check_output(commmand_to_execute, shell=True)

            print('subprocess result:')
            print(subprocess_result)
            print('decode subprocess result:')
            print(subprocess_result.decode('utf-8'))
            
            print(model +  ' [folder: ' + folder + '] ' + ' [threshold: ' + str(threshold) + ']: DONE!')