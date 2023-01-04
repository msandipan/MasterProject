import functools
from torchvision import models
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
import timm


#def efficientnet_params(model_name):    """ Map EfficientNet model name to parameter coefficients. """    params_dict = {        # Coefficients:   width,depth,res,dropout        'efficientnet-b0': (1.0, 1.0, 224, 0.2),        'efficientnet-b1': (1.0, 1.1, 240, 0.2),        'efficientnet-b2': (1.1, 1.2, 260, 0.3),        'efficientnet-b3': (1.2, 1.4, 300, 0.3),        'efficientnet-b4': (1.4, 1.8, 380, 0.4),        'efficientnet-b5': (1.6, 2.2, 456, 0.4),        'efficientnet-b6': (1.8, 2.6, 528, 0.5),        'efficientnet-b7': (2.0, 3.1, 600, 0.5),        'efficientnet-b8': (2.2, 3.6, 672, 0.5),        'efficientnet-l2': (4.3, 5.3, 800, 0.5),    }    return params_dict[model_name]

model_map = {'Dense121': models.densenet121(pretrained=True),
             'Dense121_NoPre': models.densenet121(pretrained=False),
             'VGG16BN': models.vgg16_bn(pretrained=True),
             'VGG16BN_NoPre': models.vgg16_bn(pretrained=False),
             'AlexNet':models.alexnet(pretrained=True),
             'efficientnet_b0': EfficientNet.from_pretrained('efficientnet-b0'),
             #'efficientnet_b0_NoPre': EfficientNet.from_name('efficientnet-b0'),
             #'efficientnet_b1': EfficientNet.from_pretrained('efficientnet-b1'),
             #'efficientnet_b2': EfficientNet.from_pretrained('efficientnet-b2'),
             #'efficientnet_b3': EfficientNet.from_pretrained('efficientnet-b3'),
             #'efficientnet_b3_NoPre': EfficientNet.from_name('efficientnet-b3'),
             #'efficientnet_b4': EfficientNet.from_pretrained('efficientnet-b4'),
             'efficientnet_b5': EfficientNet.from_pretrained('efficientnet-b5'),
             'efficientnet_b5_NoPre': EfficientNet.from_name('efficientnet-b5'),
             #'efficientnet_b6': EfficientNet.from_pretrained('efficientnet-b6'),
             #'efficientnet_b7': EfficientNet.from_pretrained('efficientnet-b7'),
             'Resnet50': pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet'),
             'googlenet':  models.googlenet(pretrained=True),
             #'vit_base_patch16_384': timm.create_model('vit_base_patch16_384', pretrained=True),

               }

# model_map = {'Dense121' : models.densenet121(pretrained=True),
#              'Dense121_NoPre' : models.densenet121(pretrained=False),
#              'Dense169' : models.densenet169(pretrained=True),
                #'googlenet' :  googlenet = models.googlenet(pretrained=True),
#              'Dense161' : models.densenet161(pretrained=True),
#              'Dense201' : models.densenet201(pretrained=True),
#              'VGG16BN' : models.vgg16_bn(pretrained=True),
#              'VGG16BN_noPre' : models.vgg16_bn(pretrained=False),
#              'Inception': models.inception_v3(pretrained=True),
#              'Inception_noPre': models.inception_v3(pretrained=False),
#              'Squeezenet' : models.squeezenet1_1(pretrained=True),
#              'Squeezenet_noPre' : models.squeezenet1_1(pretrained=False),
#              'Resnet50' : pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet'),
#              'Resnet101' : models.resnet101(pretrained=True),
#              'InceptionV3': pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet'),# models.inception_v3(pretrained=True),
#              'se_resnext50': pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet'),
#              'se_resnext50_noPre': pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=None),
#              'se_resnext101': pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet'),
#              'se_resnet50': pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet'),
#              'se_resnet101': pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet'),
#              'se_resnet152': pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained='imagenet'),
#              'resnext101': pretrainedmodels.__dict__['resnext101_32x4d'](num_classes=1000, pretrained='imagenet'),
#              'resnext101_64': pretrainedmodels.__dict__['resnext101_64x4d'](num_classes=1000, pretrained='imagenet'),
#              'senet154': pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet'),
#              'polynet': pretrainedmodels.__dict__['polynet'](num_classes=1000, pretrained='imagenet'),
#              'dpn92': pretrainedmodels.__dict__['dpn92'](num_classes=1000, pretrained='imagenet+5k'),
#              'dpn68b': pretrainedmodels.__dict__['dpn68b'](num_classes=1000, pretrained='imagenet+5k'),
#              'nasnetamobile': pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained='imagenet'),
#              'efficientnet_b1': EfficientNet.from_pretrained('efficientnet-b1'),#,num_classes=config['numClasses'])
#              'efficientnet_b2': EfficientNet.from_pretrained('efficientnet-b2'),#,num_classes=config['numClasses'])
#              'efficientnet_b3': EfficientNet.from_pretrained('efficientnet-b3'),#,num_classes=config['numClasses'])
#              'efficientnet_b4': EfficientNet.from_pretrained('efficientnet-b4'),#,num_classes=config['numClasses'])
#              'efficientnet_b5': EfficientNet.from_pretrained('efficientnet-b5'),#,num_classes=config['numClasses'])
#              'efficientnet_b6': EfficientNet.from_pretrained('efficientnet-b6'),#,num_classes=config['numClasses'])
#              'efficientnet_b7': EfficientNet.from_pretrained('efficientnet-b7'), #,num_classes=config['numClasses'])
#              'efficientnet_b0': EfficientNet.from_pretrained('efficientnet-b0') #,num_classes=config['numClasses'])
#                }

def getModel(model_name):
  """Returns a function for a model
  Args:
    mdlParams: dictionary, contains configuration
    is_training: bool, indicates whether training is active
  Returns:
    model: A function that builds the desired model
  Raises:
    ValueError: If model name is not recognized.
  """
  if model_name not in model_map:
    raise ValueError('Name of model unknown %s' % model_name)
  func = model_map[model_name]
  @functools.wraps(func)
  def model():
      return func
  return model
