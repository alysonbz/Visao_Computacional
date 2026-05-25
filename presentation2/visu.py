import SimpleITK as sitk
import numpy as np
import vtk
from vtk.util import numpy_support





# CONFIGURAÇÕES


input_file = "/home/ana/PycharmProjects/Visao_Computacional/presentation2/volume.mhd"



# LER VOLUME ORIGINAL


img = sitk.ReadImage(input_file)
volume = sitk.GetArrayFromImage(img)

print("Shape:", volume.shape)
print("Min:", volume.min())
print("Max:", volume.max())



# CARREGAR OU GERAR MÁSCARA


from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes

mask_final = np.zeros_like(volume, dtype=np.uint8)

for i in range(volume.shape[0]):
    slice_img = volume[i]

    mask = np.logical_and(slice_img >= -1000, slice_img <= -300)

    mask = clear_border(mask)

    labeled = label(mask)
    regions = regionprops(labeled)

    if len(regions) == 0:
        continue

    regions = sorted(regions, key=lambda r: r.area, reverse=True)

    mask_componentes = np.zeros_like(mask, dtype=np.uint8)

    for r in regions[:2]:
        mask_componentes[labeled == r.label] = 1

    mask_fechada = closing(mask_componentes, disk(5))

    mask_preenchida = binary_fill_holes(mask_fechada)

    mask_final[i] = mask_preenchida.astype(np.uint8)


print("Valores únicos da máscara:", np.unique(mask_final))
print("Soma da máscara:", np.sum(mask_final))

from skimage.measure import label, regionprops

# Componentes conectados em 3D
labeled_3d = label(mask_final, connectivity=1)
regions_3d = regionprops(labeled_3d)

print("Componentes 3D encontrados:", len(regions_3d))

regions_3d = sorted(regions_3d, key=lambda r: r.area, reverse=True)

mask_limpa_3d = np.zeros_like(mask_final, dtype=np.uint8)

# Mantém só os 2 maiores componentes 3D: pulmão esquerdo e direito
for r in regions_3d:
    z_min, y_min, x_min, z_max, y_max, x_max = r.bbox

    altura_z = z_max - z_min
    area = r.area

    print("Componente:", r.label, "Área:", area, "BBox:", r.bbox)

    # Mantém apenas componentes grandes e que ocupam várias fatias
    if area > 100000 and altura_z > 20:
        mask_limpa_3d[labeled_3d == r.label] = 1

print("Soma antes:", np.sum(mask_final))
print("Soma depois:", np.sum(mask_limpa_3d))

mask_final = mask_limpa_3d

# NUMPY PARA VTK


mask_np = mask_final.astype(np.uint8)

depth, height, width = mask_np.shape

vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(width, height, depth)
vtk_image.SetSpacing(img.GetSpacing())

flat_data = mask_np.ravel(order="C")

vtk_data = numpy_support.numpy_to_vtk(
    num_array=flat_data,
    deep=True,
    array_type=vtk.VTK_UNSIGNED_CHAR
)

vtk_image.GetPointData().SetScalars(vtk_data)


# MARCHING CUBES

surface = vtk.vtkMarchingCubes()
surface.SetInputData(vtk_image)
surface.SetValue(0, 0.5)
surface.Update()

print("Pontos gerados:", surface.GetOutput().GetNumberOfPoints())

if surface.GetOutput().GetNumberOfPoints() == 0:
    raise ValueError("Marching Cubes não gerou superfície.")



# SUAVIZAÇÃO
smooth = vtk.vtkSmoothPolyDataFilter()
smooth.SetInputConnection(surface.GetOutputPort())
smooth.SetNumberOfIterations(30)
smooth.SetRelaxationFactor(0.1)
smooth.FeatureEdgeSmoothingOff()
smooth.BoundarySmoothingOn()
smooth.Update()


# NORMAIS

normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(smooth.GetOutputPort())
normals.ComputePointNormalsOn()
normals.ComputeCellNormalsOn()
normals.Update()


# RENDERIZAÇÃO

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