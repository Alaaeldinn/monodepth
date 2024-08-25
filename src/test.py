import math

def calculate_distance(focal_length, area, calibration_constant):
    """
    Calculate the distance from the camera to a polygon in the image.
    
    Parameters:
    focal_length (float): The focal length of the camera in mm.
    area (float): The area of the polygon in pixels^2.
    calibration_constant (float): The calibration constant determined experimentally.
    
    Returns:
    float: The distance from the camera to the polygon in mm.
    """
    distance = (calibration_constant * focal_length) / math.sqrt(area)
    return distance

# Example usage:
focal_length = 50  # in mm
area = 2000  # in pixels^2
calibration_constant = 1000  # experimentally determined

distance = calculate_distance(focal_length, area, calibration_constant)
print(f"Distance to the polygon: {distance} mm")
