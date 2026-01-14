# from plyfile import PlyData, PlyElement
# import numpy as np

# === PLACEHOLDER PATHS ===
# Sostituisci questi con i tuoi file
PLY_FILE_1 = 'C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gaussian-splatting-second-pass\\output\\paper_I3D\\fields_80-20-eval-NO_EXP-shell_from_70_to_200-2ndPass\\point_cloud\\iteration_30000\\point_cloud.ply'  # File PLY second pass
PLY_FILE_2 = 'C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gass-splat-first-pass-multiply\\output\\paper_I3D\\fields_80-20-eval-NO_EXP-shell_from_70_to_200-1stPass\\point_cloud\\iteration_30000\\point_cloud.ply' # File PLY first pass
PLY_FILE_OUT = 'C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gaussian-splatting-second-pass\\output\\paper_I3D\\fields_80-20-eval-NO_EXP-shell_from_70_to_200-2ndPass\\point_cloud_30000_no_bg.ply' # File PLY risultante

# === FUNZIONE PRINCIPALE ===
def main():
	from plyfile import PlyData, PlyElement
	import numpy as np

	print(f"Loading first PLY file: {PLY_FILE_1}")
	ply1 = PlyData.read(PLY_FILE_1)
	print(f"Loading second PLY file: {PLY_FILE_2}")
	ply2 = PlyData.read(PLY_FILE_2)

	print("Extracting gaussians from both files...")
	gauss1 = np.array(ply1.elements[0].data.tolist())
	gauss2 = np.array(ply2.elements[0].data.tolist())
	print(f"Number of gaussians in first file: {len(gauss1)}")
	print(f"Number of gaussians in second file: {len(gauss2)}")


	print("Converting gaussians to tuples for fast comparison...")
	gauss1_tuples = [tuple(row) for row in gauss1]
	gauss2_tuples = set(tuple(row) for row in gauss2)

	print("Filtering gaussians using set difference...")
	filtered_tuples = [g for g in gauss1_tuples if g not in gauss2_tuples]
	print(f"Number of gaussians after filtering: {len(filtered_tuples)}")



	# Convert back to numpy structured array (1D) with original field names
	dtype = ply1.elements[0].data.dtype
	names = dtype.names
	if len(filtered_tuples) > 0:
		gauss_out = np.array(filtered_tuples, dtype=dtype)
	else:
		gauss_out = np.empty(0, dtype=dtype)

	print(f"Writing output PLY file (binary): {PLY_FILE_OUT}")
	el = PlyElement.describe(gauss_out, ply1.elements[0].name)
	PlyData([el], text=False).write(PLY_FILE_OUT)
	print("Done.")

if __name__ == '__main__':
	main()
