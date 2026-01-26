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


def obtain_lazy_view_from_the_zarr_path(input_path, scale_level, list_of_coords_for_non_tzyx_dims):
    """
    scale_level = 0 means the finest/highest (spatial) resolution, the "bottom of a pyramid"
    """
    import ngff_zarr as nz
    zarr_handle = nz.from_ngff_zarr(input_path)

    if scale_level < 0 or scale_level >= len(zarr_handle.images):
        flag_error_and_quit("scale index negative or larger than number(-1) of available resolutions that the zarr dataset offers")

    zarr_image = zarr_handle.images[scale_level]
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


def segmentation(view_into_raw_data, tracking_options = default_tracking_options):
    """
    Input ('view_into_raw_data') must be t,z,y,x even for 2D+t images.
    Output is a (possibly very large!) numpy with segmentation masks,
    and the corresponding (view into) into the 'view_into_raw_data'.
    Both outputs are (down-)scaled (given 'tracking_options') already!

    Check the 'default_tracking_options' dictionary to see what all keys are supported.
    """
    from cellpose import models as cp3_models
    from skimage.transform import resize
    from math import ceil

    m1 = tracking_options.get('segmentation_model','cyto3')
    seg_model = cp3_models.CellposeModel(model_type=m1)

    # just FYI
    do_3D = view_into_raw_data.shape[1] > 1
    print(f"seg model initiated, going to do 3D: {do_3D}")

    # figure out the (possibly) downscaled spatial size (zyx axes)
    down_scale_factors = [ \
        tracking_options.get('downscale_factor_z',1), \
        tracking_options.get('downscale_factor_y',1), \
        tracking_options.get('downscale_factor_x',1) ]
    new_spatial_size = [ ceil(size/scale) for size,scale in zip(view_into_raw_data[0].shape, down_scale_factors) ]

    # trim (along the time axis) the input data
    t_from = tracking_options.get('start_from_tp', 0)
    t_to = tracking_options.get('end_at_tp', -2) +1  # 'end_at_tp' is inclusive bound, 't_to' is exclusive, hence +1
    if t_to == -1: t_to = view_into_raw_data.shape[0]
    view_into_raw_data = view_into_raw_data[t_from:t_to]

    # 'all_masks' will be in the new downscaled size, and the trimmed length!
    all_masks = np.zeros((view_into_raw_data.shape[0],*new_spatial_size), dtype='uint16')

    for t in range(view_into_raw_data.shape[0]):
        img = np.array( resize(view_into_raw_data[t], new_spatial_size, preserve_range=True) )
        masks,_,_ = seg_model.eval([img], channels=[0,0], do_3D=do_3D, normalize=True)
        print(f"done segmenting frame {t}, input image size was {img.shape}")

        # btw, it is possible to re-use the memory into which the original zarr data landed
        #img[:] = masks[0,:]
        all_masks[t] = masks[0]

    return all_masks, view_into_raw_data


def tracking(view_into_raw_data, seg_data, tracking_options = default_tracking_options):
    """
    Both inputs ('view_into_raw_data' and 'seg_data') must be t,z,y,x even for 2D+t images,
    and of the same shapes.
    Output is that of Trackastra, and napari tracks; both with possibly downscaled spatial
    coordinates (depending on the 'tracking_options').

    Check the 'default_tracking_options' dictionary to see what all keys are supported.
    """
    from trackastra.model import Trackastra

    m2 = tracking_options.get('tracking_model','ctc')
    tra_model = Trackastra.from_pretrained(m2)

    print("tracking started...")
    track_graph = tra_model.track(view_into_raw_data, seg_data, mode="greedy")  # or mode="ilp", or "greedy_nodiv"
    print("tracking done")

    # Write to cell tracking challenge format
    # ctc_tracks, masks_tracked = graph_to_ctc(track_graph, all_masks, outdir=".")
    # nope^^^^, produces difficult (and heavy) output
    #
    # other options? CSV for Mastodon, GEFF that is natively supported with Trackastra's API
    # let's go for the CSV for Mastodon, it's pretty easy

    # TODO: upscale the coordinates in zyx axes
    # consuider also tracking_options.start_from_tp to offset the 0-based time coordinate of the 'view_into_data'


def segment_and_track_wrapper(zarr_path: str, scale_level: int, list_of_coords_for_non_tzyx_dims: list[int], tracking_options = default_tracking_options):
    """
    Check the 'default_tracking_options' dictionary to see what all keys are supported.
    It is worthwhile to downscale in x,y,z if the input images are 500+ pixels per dimension.
    """
    data_view = obtain_lazy_view_from_the_zarr_path(zarr_path, scale_level, list_of_coords_for_non_tzyx_dims)
    # NB: now the data_view is guaranteed to be order as: tzyx
    #     and it is truly an unmodified view, not scaled, not trimmed
    #
    segmentation_and_tracking(data_view, tracking_options)


def example():
    testing_zarr_path = 'https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0051/180712_H2B_22ss_Courtney1_20180712-163837_p00_c00_preview.zarr/0'
    #
    tracking_options = default_tracking_options.copy()
    tracking_options['downscale_factor_x'] = 2.0
    tracking_options['downscale_factor_y'] = 2.0
    tracking_options['downscale_factor_z'] = 1.5
    tracking_options['start_from_tp'] = 0
    tracking_options['end_at_tp'] = 5
    #
    # axes of the 'testing_zarr_path' are: 't', 'c', 'z', 'y', 'x'; and just one channel ('c')
    list_of_coords_for_non_tzyx_dims = [0] # to choose the channel
    segment_and_track_wrapper(testing_zarr_path,0, list_of_coords_for_non_tzyx_dims, tracking_options)

