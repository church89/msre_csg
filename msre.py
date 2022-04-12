import openmc
import numpy as np
import os
from math import pi, sin, cos
import pandas as pd
import matplotlib.pyplot as plt
os.system('rm *.xml *.h5')

#Define msre materials

#Fuel salt
# salt = openmc.Material()
# salt.add_nuclide('Li7', 10.9566, 'wo')
# salt.add_element('Be', 6.3492, 'wo')
# salt.add_element('Zr', 11.1013, 'wo')
# salt.add_element('Hf', 0.0001, 'wo')
# salt.add_nuclide('U234', 0.0144, 'wo')
# salt.add_nuclide('U235', 1.4093, 'wo')
# salt.add_nuclide('U236', 0.0059, 'wo')
# salt.add_nuclide('U238', 3.0652, 'wo')
# salt.add_element('Fe', 0.0162, 'wo')
# salt.add_element('Cr', 0.0028, 'wo')
# salt.add_element('Ni', 0.0030, 'wo')
# salt.add_element('O', 0.0490, 'wo')
# salt.add_element('F', 67.0270, 'wo')
# salt.set_density('g/cm3', 2.3223)

Li = openmc.Material(name='Li', temperature = 903)
Li.add_nuclide('Li7',1)
Li.set_density('g/cm3',2.3275)

Be = openmc.Material(name='Be', temperature = 903)
Be.add_element('Be',1)
Be.set_density('g/cm3',2.3275)

Zr = openmc.Material(name='Zr', temperature = 903)
Zr.add_element('Zr',1)
Zr.set_density('g/cm3',2.3275)

Hf = openmc.Material(name='Hf', temperature = 903)
Hf.add_element('Hf',1)
Hf.set_density('g/cm3',2.3275)

U234 = openmc.Material(name='U234', temperature = 903)
U234.add_nuclide('U234',1)
U234.set_density('g/cm3',2.3275)

U235 = openmc.Material(name='U235', temperature = 903)
U235.add_nuclide('U235',1)
U235.set_density('g/cm3',2.3275)

U236 = openmc.Material(name='U236', temperature = 903)
U236.add_nuclide('U236',1)
U236.set_density('g/cm3',2.3275)

U238 = openmc.Material(name='U238', temperature = 903)
U238.add_nuclide('U238',1)
U238.set_density('g/cm3',2.3275)

Fe = openmc.Material(name='Fe', temperature = 903)
Fe.add_element('Fe',1)
Fe.set_density('g/cm3',2.3275)

Cr = openmc.Material(name='Cr', temperature = 903)
Cr.add_element('Cr',1)
Cr.set_density('g/cm3',2.3275)

Ni = openmc.Material(name='Ni', temperature = 903)
Ni.add_element('Ni',1)
Ni.set_density('g/cm3',2.3275)

O = openmc.Material(name='O', temperature = 903)
O.add_element('O',1)
O.set_density('g/cm3',2.3275)

F = openmc.Material(name='F', temperature = 903)
F.add_element('F',1)
F.set_density('g/cm3',2.3275)

salt = openmc.Material.mix_materials(
    [Li, Be, Zr, Hf, U234, U235, U236, U238, Fe, Cr, Ni, O, F],
    [10.9566/100, 6.3492/100, 11.1013/100, 0.0001/100, 0.0144/100, 1.4093/100, 0.0059/100, 3.0652/100, 0.0162/100, 0.0028/100, 0.0030/100, 0.0490/100, 67.0270/100],
    'wo')
salt.name="salt"
#Control rod Gd2O3-Al2O3
Gd2O3 = openmc.Material()
Gd2O3.add_element('Gd',2)
Gd2O3.add_element('O',3)
Gd2O3.set_density('g/cm3',5.873)
Al2O3 = openmc.Material()
Al2O3.add_element('Al',2)
Al2O3.add_element('O',3)
Al2O3.set_density('g/cm3',5.873)
poison = openmc.Material.mix_materials([Gd2O3,Al2O3],[0.7,0.3],'wo')
#inor vessel
inor = openmc.Material()
inor.set_density('g/cm3',8.7745)
inor.add_element('Ni',(66+71)/2,'wo')
inor.add_element('Mo',(15+18)/2,'wo')
inor.add_element('Cr',(6+8)/2,'wo')
inor.add_element('Fe',5,'wo')
inor.add_element('C',(0.04+0.08)/2,'wo')
inor.add_element('Al',0.25,'wo')
inor.add_element('Ti',0.25,'wo')
inor.add_element('S',0.02,'wo')
inor.add_element('Mn',1.0,'wo')
inor.add_element('Si',1.0,'wo')
inor.add_element('Cu',0.35,'wo')
inor.add_element('B',0.010,'wo')
inor.add_element('W',0.5,'wo')
inor.add_element('P',0.015,'wo')
inor.add_element('Co',0.2,'wo')
#helium
helium = openmc.Material()
helium.add_element('He',1)
helium.set_density('g/cm3',0.00001)
#Graphite moderator
graphite = openmc.Material()
graphite.add_element('C',1)
#graphite.add_element('H',2)
#graphite.add_element('O',1)
#graphite.set_density('g/cm3',1)
graphite.set_density('g/cm3',1.86)
graphite.add_s_alpha_beta('c_Graphite')
#Water Body detector
water = openmc.Material()
water.add_element('H',2)
water.add_element('O',1)
water.set_density('g/cm3',1)
#Iron shield
iron = openmc.Material()
iron.add_element('Fe',1)
iron.set_density('g/cm3',7.874)
#Gadolinium
gad = openmc.Material()
gad.add_element('Gd',1)
gad.set_density('g/cm3',7.9)
#Borno Carbide
boron = openmc.Material()
boron.add_element('B',4)
boron.add_element('C',1)
boron.set_density('g/cm3',2.52)
#iron = openmc.Material.mix_materials([fe,gad],[0.99,0.01],'wo')
#Export material
material = openmc.Materials([salt,inor,graphite,poison,helium,water,iron,boron,gad])
#material.cross_sections = "/home/lorenzo/Downloads/endfb80/endfb80_hdf5/cross_sections.xml"
material.export_to_xml()


#Define msre core geometry in cm
core_height = 200
core_radius = 70
fuel_radius = 1.2645 #fuel channel equivalent radius
fuel_channels = 720 #total number of fuel channels
fuel_pattern = 6 # number or channels in the first ring, then it grows linearly
control_rod_radius = 1.27
vessel_thickness = 2.5
shield_thickness = 10
shield2_thickness = 10
detector_distance = 30 #body detector distance from reactor external surface
detector_radius = 15 #cylinder equivalent body detector radius
detector_height = 180 #cylinder equivalent body detector height
graveyard_radius = (core_radius + vessel_thickness + shield_thickness+shield2_thickness + detector_distance + detector_radius)*1.50 #add 50% to outer body

# Define surfaces and cells
z_top_in = openmc.ZPlane(z0=core_height/2)
z_bot_in = openmc.ZPlane(z0=-core_height/2)
z_top_out = openmc.ZPlane(z0=core_height/2+2*vessel_thickness,surface_id=997)
z_bot_out = openmc.ZPlane(z0=-core_height/2-2*vessel_thickness,surface_id=998)

fuel_surf = openmc.ZCylinder(r=fuel_radius)
fuel_cell = openmc.Cell(fill=salt, region = (-fuel_surf & -z_top_in & + z_bot_in))
clad_cell = openmc.Cell(fill=graphite, region = +fuel_surf)
pin_universe = openmc.Universe(cells=(fuel_cell, clad_cell))

pois_surf = openmc.ZCylinder(r=control_rod_radius)
pois_cell = openmc.Cell(fill=poison, region = (-pois_surf & -z_top_in & + z_bot_in))
outer_cell = openmc.Cell(fill=graphite, region = +pois_surf)
pois_universe = openmc.Universe(cells=(pois_cell, outer_cell))

graphite_surf = openmc.ZCylinder(r=core_radius)
graphite_cell = openmc.Cell(fill=graphite, region = (-graphite_surf & -z_top_in & + z_bot_in))
bundle_universe = openmc.Universe(cells=(graphite_cell,))

vessel_inner = openmc.ZCylinder(r=core_radius,surface_id=995)
vessel_outer = openmc.ZCylinder(r=core_radius+vessel_thickness)
shield_outer = openmc.ZCylinder(r=core_radius+vessel_thickness+shield_thickness)
shield2_outer = openmc.ZCylinder(r=core_radius+vessel_thickness+shield_thickness+shield2_thickness,surface_id=999)
graveyard_inner = openmc.Sphere (r=graveyard_radius,surface_id=996)
graveyard_outer = openmc.Sphere (r=graveyard_radius+5, boundary_type='vacuum')

body_cyl = openmc.ZCylinder(x0=core_radius + vessel_thickness + shield_thickness + shield2_thickness + detector_distance, y0=0 ,r=detector_radius)
body_top_plane = openmc.ZPlane(z0=detector_height/2)
body_bot_plane = openmc.ZPlane(z0=-detector_height/2)

rings = [i for i in range(int(fuel_channels/fuel_pattern)) if i*(i-1)==2*fuel_channels/fuel_pattern][0] # number of fuel rings
rings_array = np.arange(fuel_pattern,rings*fuel_pattern,fuel_pattern)
dist = (core_radius - fuel_radius*2*rings)/(rings + 1) #distance between fuel channesl
pitch = 2*fuel_radius + dist #fuel pitch
rings_radii = [pitch*(i+1) for i in range(len(rings_array)) ]

for i, (r,n) in enumerate(zip(rings_radii,rings_array)):
    for j in range(n):
        theta = (j/n*360)*pi/180
        x = r*cos(theta)
        y = r*sin(theta)
        pin_surf = openmc.ZCylinder(x0=x,y0=y,r=fuel_radius)
        graphite_cell.region &= + pin_surf
        pin = openmc.Cell(fill=pin_universe, region = -pin_surf)
        pin.translation = (x, y, 0)
        bundle_universe.add_cell(pin)

cr_surf = openmc.ZCylinder(x0=0,y0=0,r=control_rod_radius)
graphite_cell.region &= + cr_surf
cr = openmc.Cell(fill=pois_universe, region = -cr_surf)
bundle_universe.add_cell(cr)

core = openmc.Cell(fill=bundle_universe, region = (-vessel_inner & -z_top_in & + z_bot_in))
vessel = openmc.Cell(fill=inor, region = (+ vessel_inner & -vessel_outer & -z_top_in & + z_bot_in))
vessel_top = openmc.Cell(fill=inor, region=(-vessel_outer & +z_top_in & -z_top_out))
vessel_bot = openmc.Cell(fill=inor, region=(-vessel_outer & -z_bot_in & +z_bot_out))
shield = openmc.Cell(fill=graphite, region=(-shield_outer & +vessel_outer & -z_top_out & +z_bot_out))
shield2 = openmc.Cell(fill=iron, region=(-shield2_outer & +shield_outer & -z_top_out & +z_bot_out))
vacuum = openmc.Cell(region = -graveyard_inner & (+shield2_outer | +z_top_out | - z_bot_out) & ~(-body_cyl & -body_top_plane & +body_bot_plane))
body = openmc.Cell(fill=water, region=(-body_cyl & -body_top_plane & +body_bot_plane))
graveyard = openmc.Cell(region = (+graveyard_inner & -graveyard_outer))

root_universe = openmc.Universe(cells=[graveyard,vacuum,shield,shield2,vessel,vessel_bot,vessel_top,core,body])

geometry = openmc.Geometry(root_universe)
geometry.export_to_xml()

basis = ['xy','xz','yz']
plots = []
for base in basis:
    plot = openmc.Plot.from_geometry(geometry)
    plot.basis = base
    plot.width = (350,350)
    plot.pixels = (2000,2000)
    plot.color_by = 'material'
    plot.colors = {
        graphite: 'chocolate',
        salt: 'gold',
        inor: 'darkgrey',
        poison: 'fuchsia',
        helium: 'azure',
        water: 'blue',
        iron: 'limegreen',
        boron: 'green'
        }
    plots.append(plot)
Plots = openmc.Plots(plots)
Plots.export_to_xml()
openmc.plot_geometry()

#Define settings
settings = openmc.Settings()
settings.batches = 20
settings.inactive = 5
settings.particles = 10
source_area = openmc.stats.Box([-core_radius, -core_radius, -core_height],[ core_radius,  core_radius,  core_height],only_fissionable = True)
settings.source = openmc.Source(space=source_area)
settings.photon_transport = True
settings.surf_source_write = {
	"surface_ids": [995],
	"max_particles": 10000
}
settings.export_to_xml()

#Define tallies
tallies = openmc.Tallies()

#Flux Tally
mesh = openmc.RegularMesh()
mesh.dimension = [100,100,1]
mesh.lower_left = [-(core_radius + vessel_thickness + shield_thickness+shield2_thickness) ,-(core_radius + vessel_thickness + shield_thickness+shield2_thickness),-1]
mesh.upper_right = [(core_radius + vessel_thickness + shield_thickness+shield2_thickness) ,(core_radius + vessel_thickness + shield_thickness+shield2_thickness),0]
mesh_filter = openmc.MeshFilter(mesh)
flux_Tally = openmc.Tally(name='tally')
flux_Tally.scores = ['flux','fission','absorption']
flux_Tally.filters = [mesh_filter]
tallies.append(flux_Tally)
#Spectrum tally1
cell_filter = openmc.CellFilter(shield)
energy_bins = openmc.mgxs.GROUP_STRUCTURES["CCFE-709"]
energy_filter = openmc.EnergyFilter(energy_bins)

#Heating rate tally, required for calculating source strenght
hr_Tally = openmc.Tally(name="heating")
hr_Tally.scores = ['heating']
tallies.append(hr_Tally)

#Dose rate tally
surf_filter = openmc.SurfaceFilter(999)
detector_filter = openmc.CellFilter(body) #ste the equivalent body as detector
particles = ['photon','neutron'] #the only particles we have
for particle in particles:
    particle_filter = openmc.ParticleFilter([particle])
    energy, dose = openmc.data.dose_coefficients(particle,'AP')
    dose_filter = openmc.EnergyFunctionFilter(energy, dose)
    dose_Tally = openmc.Tally(name=particle)
    dose_Tally.scores = ['flux']
    dose_Tally.filters=[particle_filter,dose_filter,detector_filter]
    tallies.append(dose_Tally)
    current_Tally = openmc.Tally(name='current_'+particle)
    current_Tally.scores = ['current']
    current_Tally.filters = [surf_filter]
    tallies.append(current_Tally)
    spectrum_Tally = openmc.Tally(name=particle+"_spectra")
    spectrum_Tally.scores = ["flux"]
    spectrum_Tally.filters.append(cell_filter)
    spectrum_Tally.filters.append(energy_filter)
    spectrum_Tally.filters.append(particle_filter)
    tallies.append(spectrum_Tally)

tallies.export_to_xml()

model = openmc.model.Model(geometry, material, settings, tallies)
sp_filename = model.run()

#Define post-processing
results = openmc.StatePoint(sp_filename)

#Print abosorption contour
tally=results.get_tally(name='tally')
df = tally.get_pandas_dataframe(nuclides=False)
absorption = df[df['score'] == 'flux']
mean = absorption['mean'].values.reshape((100,100))
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax = plt.imshow(mean, interpolation='nearest')
plt.title("Absorption")
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig("flux.png",dpi=300)

# Print energy spectra
from spectrum_plotter import plot_spectrum_from_tally
tally1 = results.get_tally(name="neutron_spectra")
tally2 = results.get_tally(name="photon_spectra")
# plotly style
test_plot = plot_spectrum_from_tally(
    spectrum={"neutron spectra": tally1, "photon_spectra": tally2},
    x_label="Energy [eV]",
    y_label="Flux [n/cm^2s]",
    x_scale="log",
    y_scale="log",
    title="example plot 1",
    legend=True,
    plotting_package="matplotlib",
    filename="example_spectra_from_tally_plotly.png"
    #required_units="meters / source_particle",
)

#Calculate dose rates
heating_rate = results.get_tally(name='heating').mean.mean()
print("Heating rate: {}".format(heating_rate))
body_vol = pi*(detector_radius**2)*detector_height
strength = 100/(1.602*10**(-19) * heating_rate)     # Source strenght in [source-particle/sec], considering 100 MW reactor
#tot_dose = 0
for particle in particles:
    tot_dose = results.get_tally(name=particle).mean.mean()            #Effective dose in [pSv*cm3/source-particle]
    eff_dose = tot_dose/body_vol*strength*(10**(-12))/(10**(-3))*3600       # Effective dose in [mSv/h]
    print("Total body {} effective dose: {} mSv/h".format(particle,eff_dose))
    current = results.get_tally(name='current_'+particle).mean.mean()
    print("{} current: {} ".format(particle,current))
