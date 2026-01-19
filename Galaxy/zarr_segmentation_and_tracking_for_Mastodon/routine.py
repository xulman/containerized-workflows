import ngff_zarr as nz
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
