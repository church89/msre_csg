import openmc
import numpy as np
import os
from math import pi, sin, cos

os.system('rm *.xml *.h5')

#Define msre materials

#Fuel salt
salt = openmc.Material()
salt.add_nuclide('Li7', 10.9566, 'wo')
salt.add_element('Be', 6.3492, 'wo')
salt.add_element('Zr', 11.1013, 'wo')
salt.add_element('Hf', 0.0001, 'wo')
salt.add_nuclide('U234', 0.0144, 'wo')
salt.add_nuclide('U235', 1.4093, 'wo')
salt.add_nuclide('U236', 0.0059, 'wo')
salt.add_nuclide('U238', 3.0652, 'wo')
salt.add_element('Fe', 0.0162, 'wo')
salt.add_element('Cr', 0.0028, 'wo')
salt.add_element('Ni', 0.0030, 'wo')
salt.add_element('O', 0.0490, 'wo')
salt.add_element('F', 67.0270, 'wo')
salt.set_density('g/cm3', 2.3223)
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
graphite.set_density('g/cm3',1.86)
graphite.add_s_alpha_beta('c_Graphite')
#Water Body detector
water = openmc.Material()
water.add_element('H',2)
water.add_element('O',1)
water.set_density('g/cm3',1)
#Iron shield
fe = openmc.Material()
fe.add_element('Fe',1)
fe.set_density('g/cm3',7.874)
gad = openmc.Material()
gad.add_element('Gd',1)
gad.set_density('g/cm3',7.9)
iron = openmc.Material.mix_materials([fe,gad],[0.99,0.01],'wo')
#Export material
material = openmc.Materials([salt,inor,graphite,poison,helium,water,iron])
#material.cross_sections = "/home/lorenzo/Downloads/endfb80/endfb80_hdf5/cross_sections.xml"
material.export_to_xml()

#Define msre core geometry in cm
core_height = 200
core_radius = 70
fuel_radius = 1.2645 #fuel channel equivalent radius
fuel_channels = 120 #total number of fuel channels
fuel_pattern = 8 # number or channels in the first ring, then it grows linearly
control_rod_radius = 1.27
vessel_thickness = 2
shield_thickness = 10
detector_distance = 30 #body detector distance from reactor external surface
detector_radius = 15 #cylinder equivalent body detector radius
detector_height = 180 #cylinder equivalent body detector height
graveyard_radius = (core_radius + vessel_thickness + shield_thickness + detector_distance + detector_radius)*1.50 #add 50% to outer body

# Define surfaces and cells
z_top_in = openmc.ZPlane(z0=core_height/2)
z_bot_in = openmc.ZPlane(z0=-core_height/2)
z_top_out = openmc.ZPlane(z0=core_height/2+vessel_thickness)
z_bot_out = openmc.ZPlane(z0=-core_height/2-vessel_thickness)

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

vessel_inner = openmc.ZCylinder(r=core_radius)
vessel_outer = openmc.ZCylinder(r=core_radius+vessel_thickness,surface_id=998)
shield_outer = openmc.ZCylinder(r=core_radius+vessel_thickness+shield_thickness,surface_id=999)
graveyard_inner = openmc.Sphere (r=graveyard_radius)
graveyard_outer = openmc.Sphere (r=graveyard_radius+5, boundary_type='vacuum')

body_cyl = openmc.ZCylinder(x0=core_radius + vessel_thickness + shield_thickness + detector_distance, y0=0 ,r=detector_radius)
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

core = openmc.Cell(region = (-vessel_inner & -z_top_in & + z_bot_in))
vessel = openmc.Cell(fill=inor, region = (+ vessel_inner & -vessel_outer & -z_top_in & + z_bot_in))
vessel_top = openmc.Cell(fill=inor, region=(-vessel_outer & +z_top_in & -z_top_out))
vessel_bot = openmc.Cell(fill=inor, region=(-vessel_outer & -z_bot_in & +z_bot_out))
shield = openmc.Cell(fill=iron, region=(-shield_outer & +vessel_outer & -z_top_out & +z_bot_out))
vacuum = openmc.Cell(region = -graveyard_inner & (+shield_outer | +z_top_out | - z_bot_out) & ~(-body_cyl & -body_top_plane & +body_bot_plane))
body = openmc.Cell(fill=water, region=(-body_cyl & -body_top_plane & +body_bot_plane))
graveyard = openmc.Cell(region = (+graveyard_inner & -graveyard_outer))

root_universe = openmc.Universe(cells=[graveyard,vacuum,shield,vessel,vessel_bot,vessel_top,core,body])

geometry = openmc.Geometry(root_universe)
geometry.export_to_xml()

basis = ['xy','xz','yz']
plots = []
for base in basis:
    plot = openmc.Plot.from_geometry(geometry)
    plot.basis = base
    plot.width = (300,300)
    plot.pixels = (2000,2000)
    plot.color_by = 'material'
    plot.colors = {
        graphite: 'chocolate',
        salt: 'gold',
        inor: 'darkgrey',
        poison: 'fuchsia',
        helium: 'azure',
        water: 'blue',
        iron: 'limegreen'
        }
    plots.append(plot)
Plots = openmc.Plots(plots)
Plots.export_to_xml()
openmc.plot_geometry()

#Define settings
settings = openmc.Settings()
settings.batches = 100
settings.inactive = 10
settings.particles = 100000
#source_area = openmc.stats.Box([-core_radius, -core_radius, -core_height],[ core_radius,  core_radius,  core_height],only_fissionable = True)
#settings.source = openmc.Source(space=source_area)
settings.photon_transport = True
#settings.surf_source_write = {
#	"surface_ids": [999],
#	"max_particles": 10000
#}
settings.run_mode = 'fixed source'
settings.surf_source_read = {'path': 'source/surface_source.h5'}
settings.export_to_xml()

#Define tallies
tallies = openmc.Tallies()

#Flux Tally
#mesh = openmc.RegularMesh()
#mesh.dimension = [100,100,1]
#mesh.lower_left = [-core_radius,-core_radius,-1]
#mesh.upper_right = [core_radius,core_radius,0]
#mesh_filter = openmc.MeshFilter(mesh)
#flux_Tally = openmc.Tally(name='flux')
#flux_Tally.scores = ['flux','fission']
#flux_Tally.filters = [mesh_filter]
#tallies.append(flux_Tally)

#Heating rate tally, required for calculating source strenght
hr_Tally = openmc.Tally(name="heating")
hr_Tally.scores = ['heating']
tallies.append(hr_Tally)

#Dose rate tally
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

tallies.export_to_xml()

model = openmc.model.Model(geometry, material, settings, tallies)
sp_filename = model.run()

#Define post-processing
results = openmc.StatePoint(sp_filename)

# Create log-spaced energy bins from 1 keV to 10 MeV
energy_bins = np.logspace(3,7)
# Calculate pdf for source energies
probability, bin_edges = np.histogram(results.source['E'], energy_bins, density=True)
# Make sure integrating the PDF gives us unity
print(sum(probability*np.diff(energy_bins)))
# Plot source energy PDF
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax = plt.semilogx(energy_bins[:-1], probability*np.diff(energy_bins), drawstyle='steps')
plt.xlabel('Energy (eV)')
plt.ylabel('Probability/eV')
plt.savefig("Source_energy_distribution.png",dpi=300)

heating_rate = results.get_tally(name='heating').mean.mean()
body_vol = pi*(detector_radius**2)*detector_height
strength = 100/(1.602*10**(-19) * 72739444.92723805)*  0.12486587784647982

tot_dose = 0
for particle in particles:
    tot_dose = results.get_tally(name=particle).mean.mean()            #Effective dose in [pSv*cm3/source-particle]
    eff_dose = tot_dose/body_vol*strength*(10**(-12))/(10**(-3))*3600       # Effective dose in [mSv/h]
    print("Total {} body absorbed effective dose: {} mSv/h".format(particle,eff_dose))
