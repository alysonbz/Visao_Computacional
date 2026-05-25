import SimpleITK as sitk
import numpy as np
import vtk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from vtk.util import numpy_support


input_file = "/presentation2/volume.mhd"

img = sitk.ReadImage(input_file)
volume = sitk.GetArrayFromImage(img)

print("Shape:", volume.shape)
print("Min:", volume.min(), "Max:", volume.max())


mask_final = np.zeros_like(volume, dtype=np.uint8)

for i in range(volume.shape[0]):
    slice_img = volume[i]

    binary = np.logical_and(slice_img >= -1000, slice_img <= -300)

    binary = clear_border(binary)

    labeled = label(binary)
    regions = regionprops(labeled)

    if len(regions) == 0:
        continue

    regions = sorted(regions, key=lambda r: r.area, reverse=True)

    slice_mask = np.zeros_like(binary, dtype=np.uint8)

    for r in regions[:2]:
        slice_mask[labeled == r.label] = 1

    mask_final[i] = slice_mask


print("Valores únicos após segmentação:", np.unique(mask_final))
print("Soma da máscara:", np.sum(mask_final))

if len(np.unique(mask_final)) < 2:
    raise ValueError("A máscara precisa ter 0 e 1.")


# ==========================
# NUMPY PARA VTK
# ==========================

mask_np = mask_final.astype(np.uint8)

depth, height, width = mask_np.shape

vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(width, height, depth)

flat_data = mask_np.ravel(order="C")
vtk_data = numpy_support.numpy_to_vtk(
    num_array=flat_data,
    deep=True,
    array_type=vtk.VTK_UNSIGNED_CHAR
)

vtk_image.GetPointData().SetScalars(vtk_data)


# ==========================
# MARCHING CUBES
# ==========================

surface = vtk.vtkMarchingCubes()
surface.SetInputData(vtk_image)
surface.SetValue(0, 0.5)
surface.Update()

print("Pontos gerados:", surface.GetOutput().GetNumberOfPoints())

if surface.GetOutput().GetNumberOfPoints() == 0:
    raise ValueError("MarchingCubes não gerou superfície.")


# ==========================
# SUAVIZAÇÃO DA SUPERFÍCIE
# ==========================

smooth = vtk.vtkSmoothPolyDataFilter()
smooth.SetInputConnection(surface.GetOutputPort())
smooth.SetNumberOfIterations(50)
smooth.SetRelaxationFactor(0.1)
smooth.FeatureEdgeSmoothingOff()
smooth.BoundarySmoothingOn()
smooth.Update()

normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(smooth.GetOutputPort())
normals.ComputePointNormalsOn()
normals.ComputeCellNormalsOn()
normals.Update()


# ==========================
# RENDERIZAÇÃO
# ==========================

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(normals.GetOutputPort())
mapper.ScalarVisibilityOff()

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.85, 0.85, 0.85)
actor.GetProperty().SetSpecular(0.3)
actor.GetProperty().SetSpecularPower(20)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.2)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(900, 900)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

renderer.ResetCamera()
render_window.Render()
interactor.Start()