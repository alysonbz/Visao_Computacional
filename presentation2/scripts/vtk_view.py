import vtk

filename = "../rt6.mhd"

reader = vtk.vtkMetaImageReader()
reader.SetFileName(filename)
reader.Update()

# Mapeamento do volume para visualização
volume_mapper = vtk.vtkSmartVolumeMapper()
volume_mapper.SetInputConnection(reader.GetOutputPort())

# Propriedades de opacidade e cor
volume_color = vtk.vtkColorTransferFunction()
volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
volume_color.AddRGBPoint(255, 1.0, 1.0, 1.0)

volume_scalar_opacity = vtk.vtkPiecewiseFunction()
volume_scalar_opacity.AddPoint(0, 0.0)
volume_scalar_opacity.AddPoint(255, 1.0)

volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(volume_color)
volume_property.SetScalarOpacity(volume_scalar_opacity)
volume_property.ShadeOff()
volume_property.SetInterpolationTypeToLinear()

volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

renderer = vtk.vtkRenderer()
renderer.AddVolume(volume)
renderer.SetBackground(0.1, 0.1, 0.2)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 800)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Melhorias: resetar câmera e iniciar renderização
renderer.ResetCamera()
render_window.Render()
interactor.Start()