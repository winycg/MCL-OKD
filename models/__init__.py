
'''baseline'''
from .resnet import resnet32, resnet110
from .vgg import vgg16
from .resnet_imagenet import resnet18, resnet34
from .densenet import densenetd40k12
from .hcgnet import hcgnet_A1

'''DML'''
from .dml_densenet import dml_densenetd40k12
from .dml_resnet import dml_resnet32, dml_resnet110
from .dml_vgg import dml_vgg16
from .dml_hcgnet import dml_hcgnet_A1

'''CL-ILR'''
from .cl_ilr_densenet import cl_ilr_densenetd40k12
from .cl_ilr_resnet import cl_ilr_resnet32
from .cl_ilr_resnet import cl_ilr_resnet110
from .cl_ilr_vgg import cl_ilr_vgg16
from .cl_ilr_hcgnet import cl_ilr_hcgnet_A1


'''ONE'''
from .one_densenet import one_densenetd40k12
from .one_resnet import one_resnet32, one_resnet110
from .one_vgg import one_vgg16
from .one_hcgnet import one_hcgnet_A1

'''OKDDip'''
from .okddip_densenet import okddip_densenetd40k12
from .okddip_resnet import okddip_resnet32, okddip_resnet110
from .okddip_vgg import okddip_vgg16
from .okddip_hcgnet import okddip_hcgnet_A1


'''MCL-OKD'''
from .mcl_okd_densenet import mcl_okd_densenetd40k12
from .mcl_okd_resnet import mcl_okd_resnet32, mcl_okd_resnet110
from .mcl_okd_resnet_imagenet import mcl_okd_resnet18, mcl_okd_resnet34
from .mcl_okd_vgg import mcl_okd_vgg16
from .mcl_okd_hcgnet import mcl_okd_hcgnet_A1








