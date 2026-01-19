# INPUT PARAMETERS:
# - path to the zarr dataset
# - downscale in x,y,z, default value 1,1,1
# - channel and other axes/dimensions that would need to be pinned/fixed
# - time points span (to be able to test on a short range)
#
# OUTPUT PARAMETERS:
# - name of the .csv file into which the tracking would be saved
#   (the .csv together with the original input zarr can be opened Mastodon tracking software)
# - OPTIONAL! path to zarr into which segmentation will be saved
#
# Note: The intermediate segmentation results will not be saved for now...

from cellpose import models as cp3_models
from trackastra.model import Trackastra

device = 'cpu'
seg_model = cp3_models.CellposeModel(model_type='cyto3')
tra_model = Trackastra.from_pretrained('ctc', device='automatic')

i = nz.from_ngff_zarr('https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0051/180712_H2B_22ss_Courtney1_20180712-163837_p00_c00_preview.zarr/0')
len(i.images)
i.images[1].data.shape
i.images[1].data[10,0,100]


# masks,_,_ = seg_model.eval([img], normalize=False, channels=[0,0], do_3D=False)
def flag_error_and_quit(error_msg):
    print(f"ERROR: {error_msg}")
    # exit(0) -- something that stops the interpreter


def obtain_lazy_view_from_the_zarr_path(input_path, image_idx, list_of_coords_for_non_tzyx_dims):
    import ngff_zarr as nz
    zarr_handle = nz.from_ngff_zarr(input_path)

    if image_idx < 0 or image_idx >= len(zarr_handle.images):
        flag_error_and_quit("image index negative or larger than what the zarr dataset offers")

    zarr_image = zarr_handle.images[image_idx]
    #zarr_image.data.shape
    #zarr_image.dims

    axes_known = []
    axes_unknown = []
    curr_axis_idx = 0 #aka dim number
    for d in zarr_image.dims:
        if not d in "tzyx":
            # dimension to be "moved" to the front
            axes_unknown.append(curr_axis_idx)
        else:
            axes_known.append(curr_axis_idx)
        curr_axis_idx += 1

    axes_permutation = [*axes_unknown, *axes_known]
    view = zarr_image.data.transpose(axes_permutation)[*list_of_coords_for_non_tzyx_dims]
    if len(view.shape) != 4:
        #ds = [ zarr_image.dims[n] for n in axes_permutation[-4:] ]
        flag_error_and_quit(f"after fixing non_tzyx dimensions, tzyx (4) dimensions were supposed to left; instead {len(view.shape)} dimensions are available")

    return view

