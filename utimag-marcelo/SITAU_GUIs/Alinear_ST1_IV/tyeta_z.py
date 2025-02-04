import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

plt.ion()

# Parameters

def plot_rots(tilt_angle_x, tilt_angle_z):

    row_x_offset = 0
    element_spacing = 10  # Spacing between elements in mm
    transducer_length = 100  # Length of the transducer in mm
    num_elements = 11

    fixed_z_min = -30  # Fixed minimum z-axis limit
    fixed_z_max = 30  # Fixed maximum z-axis limit

    fixed_x_min = -60  # Fixed minimum x-axis limit
    fixed_x_max = 60  # Fixed maximum x-axis limit

    fixed_y_min = -60  # Fixed minimum y-axis limit
    fixed_y_max = 60  # Fixed maximum y-axis limit

    # Create figure and axes
    fig = plt.figure(figsize=(7, 8))
    ax1 = fig.add_subplot(111, projection='3d')

    # Define the grid points for the transducer surface
    x = np.linspace(-transducer_length / 2, transducer_length / 2, num_elements)
    y = np.linspace(-transducer_length / 2, transducer_length / 2, num_elements)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(Y.shape)

    # Flatten the arrays for easier transformation
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Combine into a single array of points
    points = np.vstack((X_flat, Y_flat, Z_flat)).T

    # Define the rotation: first tilt in x-axis, then rotate around z-axis
    rotation_x = Rotation.from_euler('x', tilt_angle_x, degrees=True)
    rotation_z = Rotation.from_euler('z', tilt_angle_z, degrees=True)

    # Apply the rotations independently
    rotated_points_x = rotation_x.apply(points)
    rotated_points = rotation_z.apply(rotated_points_x)

    # Extract the rotated coordinates
    X_rotated = rotated_points[:, 0].reshape(X.shape)
    Y_rotated = rotated_points[:, 1].reshape(Y.shape)
    Z_rotated = rotated_points[:, 2].reshape(Z.shape)

    # Plot the tilted surface
    ax1.plot_surface(X_rotated, Y_rotated, Z_rotated, color='lightgrey', alpha=0.5, edgecolor='none')

    ax1.plot_wireframe(X_rotated, Y_rotated, Z_rotated, color='black', linestyle='--', linewidth=0.5)


    mid_X = (X_rotated[:-1, :-1] + X_rotated[1:, 1:]) / 2
    mid_Y = (Y_rotated[:-1, :-1] + Y_rotated[1:, 1:]) / 2
    mid_Z = (Z_rotated[:-1, :-1] + Z_rotated[1:, 1:]) / 2


    mid_X_1 = np.zeros(mid_X.shape)
    mid_X_1[-1,:] = mid_X[-1,:]

    mid_Y_1 = np.zeros(mid_Y.shape)
    # mid_Y_1[-1,:] = mid_Y[-1,:]

    mid_Z_1 = np.zeros(mid_Z.shape)
    # mid_Z_1[-1,:] = mid_X[-1,:]

    for i in np.arange(mid_X_1.shape[0]):
        for j in np.arange(mid_X_1.shape[1]):
            print(mid_X_1[i,j])
            if not mid_X_1[i,j] == 0:
                # print(mid_X_1[i,j])
                ax1.scatter(mid_X_1[i,j], mid_Y[i,j], mid_Z[i,j], color='red', s=50, marker='o', label='Midpoints')



    # for i in np.arange(mid_X_1.shape[0]):
    #     center_index = 9
    #     el = i
    #     midpoint_x = mid_X[center_index][el]
    #     midpoint_y = mid_Y[center_index][el]
    #     midpoint_z = mid_Z[center_index][el]
    #
    #     # Plot TOF line from the midpoint to the minimum z-axis value
    #     ax1.plot(
    #         [midpoint_x, midpoint_x],  # x coordinates (start and end are the same)
    #         [midpoint_y, midpoint_y],  # y coordinates (start and end are the same)
    #         [midpoint_z, -30], # z coordinates (from midpoint to minimum z value)
    #         'g--',                     # line style: green dashed line
    #         linewidth=1                # line width
    #     )


    X_row = np.linspace(-transducer_length / 2, transducer_length / 2, num_elements)
    Y_row = np.full(num_elements, row_x_offset)
    Z_row = X_row * np.tan(tilt_angle_z)  # Tilt applied to the row

    # ax1.plot(X_rotated, Y_rotated, Z_rotated, color='r', linestyle='-', linewidth=2)
    # ax1.scatter(X_rotated, Y_rotated, Z_rotated, color='r', s=50)

    # Set axis limits
    ax1.set_zlim(fixed_z_min, fixed_z_max)
    ax1.set_xlim(fixed_x_min, fixed_x_max)
    ax1.set_ylim(fixed_y_min, fixed_y_max)

    # Axis labels
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")

    ax1.plot([fixed_x_min, fixed_x_max], [0, 0], [0, 0], 'b-', linewidth=2)
    #ax1.plot([0, 0], [0, 0], [fixed_z_min, fixed_z_max], 'b-', linewidth=2)

    # Optional: Change the perspective (view angle)
    #ax1.view_init(elev=30, azim=90)

    plt.tight_layout()

    plt.savefig('theta_z_search_2.png', bbox_inches='tight')

    plt.show()
