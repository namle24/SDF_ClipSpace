import pyvista as pv
import numpy as np
import time
import matplotlib.pyplot as plt

def look_at(eye, target):
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)   

    forward = target - eye  #n vector
    forward /= np.linalg.norm(forward)

    # Generate a random unit vector perpendicular to forward
    up = np.random.randn(3)  # Generate random vector
    up = up - np.dot(up, forward) * forward  # Make it perpendicular to forward
    up = up / np.linalg.norm(up)  # Normalize to unit length     

    right = np.cross(forward, up)  #u vector
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)
    rotation = np.array([
        [right[0],    right[1],    right[2],    0],
        [true_up[0],  true_up[1],  true_up[2],  0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0,           0,           0,           1]
    ])
    translation = np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]
    ])
    return rotation @ translation

def perspective(fov, near, far):
    cot = 1.0 / np.tan(fov / 2) 
    return np.array([
        [cot, 0, 0, 0],
        [0, cot, 0, 0],
        [0, 0, -far/(far-near), -(near*far)/(far-near)],  
        [0, 0, -1, 0]
    ])

def cal_mesh_side(mesh):
    # Calculate the bounding box of the mesh
    min_coords = np.min(mesh.points, axis=0)
    max_coords = np.max(mesh.points, axis=0)
    longest_distance = np.linalg.norm(max_coords - min_coords)    
    return min_coords, max_coords, longest_distance

def cal_SDF(mesh, face_centers, face_normals, face_id, fov = 90):
    inward_normal = -face_normals[face_id]
    eye = face_centers[face_id] + 0.001*inward_normal
    target = eye + inward_normal
    fov = np.radians(fov)
    near = 0.001
    far = 5.0 #np.floor(longest_distance) + 1

    # Get view and projection matrices
    V = look_at(eye, target)
    P = perspective(fov, near, far)
    PV = P @ V

    # Transform face centers to perspective space
    face_centers_homo = np.column_stack([face_centers, np.ones(len(face_centers))])
    face_centers_inCam = (V @ face_centers_homo.T).T  # Transform to clip space
    face_centers_inCam = face_centers_inCam[:, :3] #/ face_centers_inCam[:, 3].reshape(-1, 1)  # not need Perspective divide

    #Transform mesh to perspective space
    transformed_mesh = mesh.copy()
    vertices = transformed_mesh.points   
    vertices_homo = np.column_stack([vertices, np.ones(len(vertices))])
    vertices_clip = (PV @ vertices_homo.T).T  # Transform vertices to clip space
    vertices_clip = vertices_clip[:, :3] / vertices_clip[:, 3].reshape(-1, 1)    # Perform perspective divide and keep only xyz coordinates
    transformed_mesh.points = vertices_clip
    transformed_face_centers = transformed_mesh.cell_centers().points
    transformed_normals = transformed_mesh.compute_normals(point_normals=False, cell_normals=True, consistent_normals=True).cell_data['Normals']

    # compute triangle_bboxes
    faces = transformed_mesh.faces.reshape(-1, 4)[:, 1:]  # Reshape and remove the first column (vertex count)    
    # triangles2d = []    # Convert faces to 2D triangles using transformed vertices    
    triangle_bboxes = []
    for face in faces:
        # Get the 3 vertices of the face
        v1 = transformed_mesh.points[face[0]]
        v2 = transformed_mesh.points[face[1]]
        v3 = transformed_mesh.points[face[2]]
        # Calculate bounding box for the triangle
        min_x = min(v1[0], v2[0], v3[0])
        max_x = max(v1[0], v2[0], v3[0])
        min_y = min(v1[1], v2[1], v3[1])
        max_y = max(v1[1], v2[1], v3[1])
        # Create 2D triangle using only x,y coordinates
        triangle = np.array([
            [v1[0], v1[1]],  # x,y of first vertex
            [v2[0], v2[1]],  # x,y of second vertex
            [v3[0], v3[1]]   # x,y of third vertex
        ])
        bbox = np.array([min_x, min_y, max_x, max_y]) # Store as [min_x, min_y, max_x, max_y]
        triangle_bboxes.append(bbox)    
    triangle_bboxes = np.array(triangle_bboxes) # in perspective

    # Check if points are within the view frustum
    tan_fovy = np.tan(fov / 2)
    in_range_mask = (
        (face_centers_inCam[:, 2] < -0.005) &  # Points must be in front of camera (negative z in camera space)
        (abs(face_centers_inCam[:, 0] / face_centers_inCam[:, 2]) <= tan_fovy) &  # Within horizontal FOV
        (abs(face_centers_inCam[:, 1] / face_centers_inCam[:, 2]) <= tan_fovy)    # Within vertical FOV
    )
    # in_range_mask = (     
    #     (face_centers_inCam[:, 2] < -0. )
    #     &(0 < transformed_face_centers[:, 2]) & (transformed_face_centers[:, 2] < 1) 
        
    #     &( (np.sqrt(transformed_face_centers[:, 0]**2 + transformed_face_centers[:, 1]**2) <= 1) |  (-0.1 <face_centers_inCam[:, 2] )) # Within unit circle
    # )
    in_range_indices = np.where(in_range_mask)[0]  # Get indices of faces that satisfy the conditions

    # Define the number of rays to generate
    num_rays_per_ring = 10  # Number of rays in each ring
    num_rings = 5         # Number of concentric rings
    num_rays = num_rays_per_ring * num_rings
    # Generate regular circular distribution
    ray_directions = []
    for ring in range(num_rings):  
        # Calculate radius for this ring (from 0 to 1)
        r = (ring + 0.5) / num_rings  # Offset by 0.5 to center the rings        
        # Calculate number of points in this ring, More points in outer rings to maintain even distribution
        num_points = int(num_rays_per_ring * (ring + 1))        
        # Generate evenly spaced angles for this ring
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)        
        # Convert to Cartesian coordinates
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        z = np.ones_like(x)        
        # Scale by tan(fovy/2) to respect the FOV
        tan_fov = np.tan(fov / 2)
        x = x * tan_fov
        y = y * tan_fov        
        # Stack coordinates
        ring_directions = np.column_stack((x, y, z))
        ray_directions.append(ring_directions)
    ray_directions = np.vstack(ray_directions)
    # Iterate through all ray directions
    epsilon = (np.tan(fov/2) / num_rings)  #distance_btw_2rays =  np.tan(fovy/2)
    # Combine all rings
   

    intersect_indices = [] #Find all the first face interect with ray
    for ray_direction in ray_directions:
        in_range_bboxes = triangle_bboxes[in_range_indices]
        in_range_normals = transformed_normals[in_range_indices]
        contains_p = (
            #(in_range_normals[:, 2] <= 0) &
            (in_range_bboxes[:, 0] <= ray_direction[0]) & (ray_direction[0] <= in_range_bboxes[:, 2]) &
            (in_range_bboxes[:, 1] <= ray_direction[1]) & (ray_direction[1] <= in_range_bboxes[:, 3])
        )
        containing_faces = in_range_indices[contains_p]
        #intersect_indices.extend(containing_faces)

        if len(containing_faces) > 0:            
            matching_z = transformed_face_centers[containing_faces, 2]  # Get z-coordinates of face centers for containing faces
            min_z_index = np.argmin(matching_z)  # because the z values are negative          
            closest_index = containing_faces[min_z_index]             # Get the corresponding index from containing_faces
            # Check if normal points inward (negative z)
            #if transformed_normals[closest_index][2] <= 0:
            intersect_indices.append(closest_index)
    # Convert to numpy array and remove duplicates
    intersect_indices = np.unique(np.array(intersect_indices)) 

    #Calculate average distance from each face center to intersect indices
    if(len(intersect_indices) > 0):
        distances = np.linalg.norm(face_centers[intersect_indices] - face_centers[face_id], axis=1)   
        avg_distance = np.mean(distances)
        return avg_distance
    else:
        return 0

    # ########################################################################    
    transformed_mesh.cell_data['class'] = 0
    if(len(intersect_indices) > 0):
        transformed_mesh.cell_data['class'][intersect_indices] = 1
        distances = np.linalg.norm(face_centers[intersect_indices] - face_centers[face_id], axis=1)   
        avg_distance = np.mean(distances)
        print("avg_distance", avg_distance)
    
    print(f"Number of in-range indices: {len(in_range_indices)}")
    print(f"Number of intersect indices: {len(intersect_indices)}")  
    
    mesh.cell_data['class'] = 0
    if(len(in_range_indices) > 0):
        mesh.cell_data['class'][in_range_indices] = 1
    
    rmesh = mesh.copy()
    rmesh.cell_data['cut'] = 0
    if(len(intersect_indices) > 0):
        rmesh.cell_data['cut'][intersect_indices] = 1

    plotter = pv.Plotter(shape=(1, 2))
    plotter.subplot(0, 0)    # THE LEFT FRAME
    plotter.add_mesh(mesh, scalars='class', opacity=0.9, show_edges=True)
    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # Add coordinate axes to the plotter
    plotter.add_arrows(origin, x_axis, color='red', mag=0.3, label='X')
    plotter.add_arrows(origin, y_axis, color='green', mag=0.3, label='Y')
    plotter.add_arrows(origin, z_axis, color='blue', mag=0.3, label='Z')
    plotter.add_arrows(eye, inward_normal, color='yellow', mag=0.2, label='Inward')
    #plotter.show_grid()
    
    plotter.subplot(0, 1)   # THE RIGHT FRAME
    plotter.add_mesh(rmesh, scalars="cut", opacity=0.7, show_edges=True)
    plotter.add_arrows(origin, x_axis, color='red', mag=0.3, label='X')
    plotter.add_arrows(origin, y_axis, color='green', mag=0.3, label='Y')
    plotter.add_arrows(origin, z_axis, color='blue', mag=0.3, label='Z')
    plotter.add_arrows(eye, inward_normal, color='yellow', mag=0.2, label='Inward')  

    # Link the cameras of both subplots    # Show the plot
    #plotter.add_key_event('q', lambda: plotter.close())
    plotter.show(full_screen=True)

def main(args):
    mesh = pv.read(args.obj_file) #eg: --obj_file=./Models/158.obj
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    # #mesh = pv.read(".\\Models\\Synthetic\\boxtorus.obj") 
    print("Mesh loaded:", len(mesh.points), "vertices,", len(mesh.faces)/4, "faces")
    
    start_time = time.time()
    face_centers = mesh.cell_centers().points
    face_normals = mesh.compute_normals(point_normals=False, cell_normals=True, consistent_normals=True).cell_data['Normals']
    #_,_,longest = cal_mesh_side(mesh)

    # cal_SDF(mesh, face_centers, face_normals, 1563, 120)
    # exit()

    sdf = []
    for face_id in range(len(face_centers)):
        avg_distance = cal_SDF(mesh, face_centers, face_normals, face_id, 90)
        sdf.append(avg_distance)
        if face_id % 10 == 0:
            print(f"Face {face_id}: Average distance = {avg_distance:.4f}")
        if(avg_distance == 0):
            print(f"Face {face_id}: have no intersection")
    
    end_time = time.time()
    print(f"\n--- Thoi gian tinh toan: {end_time - start_time:.2f} giay ---")

    # Write SDF values to a text file
    import os
    output_file = os.path.splitext(args.obj_file)[0] + '.txt'
    with open(output_file, 'w') as f:
        for face_id, sdf_value in enumerate(sdf):
            f.write(f"{face_id},{sdf_value}\n")

    # # Create a new plotter
    mesh.cell_data['sdf'] = sdf
    plotter = pv.Plotter()

    # Add the mesh with SDF values as colors
    plotter.add_mesh(mesh, scalars="sdf", cmap='jet', show_edges=True)
    # Add a color bar
    #plotter.add_scalar_bar(title='SDF Values')

    # Show the plot
    plotter.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spherical SOM surface segmentation.")
    parser.add_argument("--obj_file", type=str, required=False, default=r"D:\CMSVTransformer\data\radio_0026.off", help="Path to the OBJ file")
    args = parser.parse_args()
    main(args)
