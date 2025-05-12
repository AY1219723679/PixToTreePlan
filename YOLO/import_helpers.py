"""
Helper script to ensure correct imports for the PixToTreePlan project.
Use this to import modules from the project regardless of how it is structured.
"""

import os
import sys


def setup_imports():
    """
    Set up the import paths for the PixToTreePlan project.
    This function ensures that the symbolic link between core/main and main
    is properly set up if needed.
    
    Returns:
        bool: True if the setup was successful, False otherwise.
    """
    # Get the path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the project root (parent of YOLO directory)
    project_root = os.path.dirname(script_dir)
    
    # Add the project root to the Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Check for the main directory
    main_path = os.path.join(project_root, 'main')
    if not os.path.exists(main_path):
        print(f"Warning: 'main' directory not found at {main_path}")
        return False
    
    # Check for the core directory and create symbolic link if needed
    core_dir = os.path.join(project_root, 'core')
    core_main_path = os.path.join(core_dir, 'main')
    
    if not os.path.exists(core_main_path):
        try:
            print("Setting up symbolic link between core/main and main...")
            
            # Create the core directory if it doesn't exist
            if not os.path.exists(core_dir):
                os.makedirs(core_dir, exist_ok=True)
            
            # Create the symbolic link - for Windows, use a junction
            if os.name == 'nt':  # Windows
                os.system(f'New-Item -ItemType Junction -Path "{core_main_path}" -Target "{main_path}" -Force')
            else:  # Unix-like systems
                os.symlink(main_path, core_main_path, target_is_directory=True)
                
            print(f"Created symbolic link from {core_main_path} to {main_path}")
        except Exception as e:
            print(f"Error creating symbolic link: {str(e)}")
            print("Please run the following command from the project root:")
            if os.name == 'nt':  # Windows
                print(f"New-Item -ItemType Junction -Path \"core/main\" -Target \"main\" -Force")
            else:  # Unix-like systems
                print(f"ln -s main core/main")
            return False
    
    return True


if __name__ == "__main__":
    # Test the import setup
    success = setup_imports()
    
    if success:
        print("Import paths set up successfully!")
        try:
            from main.img_to_pointcloud.coord_utils import pixel_coords_to_3d
            print("Successfully imported pixel_coords_to_3d from main.img_to_pointcloud.coord_utils")
        except ImportError:
            print("Failed to import from main.img_to_pointcloud.coord_utils")
            try:
                from core.main.img_to_pointcloud.coord_utils import pixel_coords_to_3d
                print("Successfully imported pixel_coords_to_3d from core.main.img_to_pointcloud.coord_utils")
            except ImportError:
                print("Failed to import from core.main.img_to_pointcloud.coord_utils")
    else:
        print("Failed to set up import paths!")
