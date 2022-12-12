import sys
from optparse import OptionParser
import json




parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="pascal_voc")
parser.add_option("-n", "--num_rois", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
                  default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90",
                  help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--num_epochs", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename", help=
"Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.",
                  default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path",
                  help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()



import config_train as conf

print(conf.ee)



def recordConfig_as_text(config: OptionParser(), filePath: str):
    opt = str(config)
    opt = opt.replace("'", '"')
    opt = opt.replace("None", "null")
    opt = opt.replace("False", "false")
    opt = opt.replace("True", "true")
    # opt = '{"ab": null, "bc": "123", "cc": false}'
    dictdata = json.loads(opt)
    with open(file=filePath, mode='w') as f:
        for da in dictdata:
            # print(da, ':', dictdata[da])
            text = str(da) + ' = ' + str(dictdata[da])
            f.write(text + '\n')


