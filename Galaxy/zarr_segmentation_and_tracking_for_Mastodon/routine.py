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

import numpy as np

default_tracking_options = {
    'downscale_factor_x' : 1.0,
    'downscale_factor_y' : 1.0,
    'downscale_factor_z' : 1.0,
    'start_from_tp'      : 0,
    'end_at_tp'          : -1,

    # future extension -- currently not used in the code
    'use_gpu'            : False,
    'segmentation_model' : 'cyto3',
    'tracking_model'     : 'ctc'
}



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

    if len(axes_unknown) != len(list_of_coords_for_non_tzyx_dims):
        flag_error_and_quit(f"found {len(axes_unknown)} non_tzyx dimensions but different number ({len(list_of_coords_for_non_tzyx_dims)}) of values for them")

    axes_permutation = [*axes_unknown, *axes_known]
    # NB: TODO, would be great to check the order in the 'axes_known' and possibly adjust it...
    view = zarr_image.data.transpose(axes_permutation)[*list_of_coords_for_non_tzyx_dims]

    if 'z' not in zarr_image.dims:
        # assuming then tyx, thus injecting 'z':
        view = np.reshape(view, (view.shape[0],1,view.shape[1],view.shape[2]))

    if len(view.shape) != 4:
        #ds = [ zarr_image.dims[n] for n in axes_permutation[-4:] ]
        flag_error_and_quit(f"after fixing non_tzyx dimensions, tzyx (4) dimensions were supposed to be left; instead {len(view.shape)} dimensions are available")

    return view


def segmentation_and_tracking(view_into_data, tracking_options = default_tracking_options):
    """
    Input (`view_into_data`) must be t,z,y,x even for 2D+t images.
    Output is that of Trackastra.

    Check the 'default_tracking_options' dictionary to see what all keys are supported.
    """
    from cellpose import models as cp3_models
    from trackastra.model import Trackastra
    from skimage.transform import resize
    from math import ceil

    seg_model = cp3_models.CellposeModel(model_type='cyto3') # TODO use default_tracking_options.segmentation_model
    tra_model = Trackastra.from_pretrained('ctc')

    # just FYI
    do_3D = view_into_data.shape[1] > 1
    print(f"seg and tra models initiated, going to do 3D: {do_3D}")

    # figure out the (possibly) downscaled spatial size (zyx axes)
    down_scale_factors = [ \
        tracking_options.get('downscale_factor_z',1), \
        tracking_options.get('downscale_factor_y',1), \
        tracking_options.get('downscale_factor_x',1) ]
    new_spatial_size = [ ceil(size/scale) for size,scale in zip(view_into_data[0].shape, down_scale_factors) ]

    # trim (along the time axis) the input data
    t_from = tracking_options.get('start_from_tp', 0)
    t_to = tracking_options.get('end_at_tp', -2) +1  # 'end_at_tp' is inclusive bound, 't_to' is exclusive, hence +1
    if t_to == -1: t_to = view_into_data.shape[0]
    view_into_data = view_into_data[t_from:t_to]

    # 'all_masks' will be in the new downscaled size, and the trimmed length!
    all_masks = np.zeros((view_into_data.shape[0],*new_spatial_size), dtype='uint16')

    for t in range(view_into_data.shape[0]):
        img = np.array( resize(view_into_data[t], new_spatial_size, preserve_range=True) )
        masks,_,_ = seg_model.eval([img], channels=[0,0], do_3D=do_3D, normalize=True)
        print(f"done segmenting frame {t}, input image size was {img.shape}")

        # btw, it is possible to re-use the memory into which the original zarr data landed
        #img[:] = masks[0,:]
        all_masks[t] = masks[0]

    # now the view_into_data contains the segmentation
    print("tracking started...")
    track_graph = tra_model.track(view_into_data, all_masks, mode="greedy")  # or mode="ilp", or "greedy_nodiv"
    print("tracking done")

    # Write to cell tracking challenge format
    # ctc_tracks, masks_tracked = graph_to_ctc(track_graph, all_masks, outdir=".")
    # nope^^^^, produces difficult (and heavy) output
    #
    # other options? CSV for Mastodon, GEFF that is natively supported with Trackastra's API
    # let's go for the CSV for Mastodon, it's pretty easy

    # TODO: upscale the coordinates in zyx axes
    # consuider also tracking_options.start_from_tp to offset the 0-based time coordinate of the 'view_into_data'


