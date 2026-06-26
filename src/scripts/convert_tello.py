import omni.kit.app
from isaacsim.ros2.urdf import _urdf_importer # Adjust import based on Isaac Sim version

def convert_urdf_to_usd(urdf_path, dest_usd_path):
    # Initialize the URDF importer extension settings
    status, import_config = _urdf_importer.acquire_urdf_importer_interface().create_import_config()
    
    # Configure settings for a drone
    import_config.merge_fixed_joints = False
    import_config.fix_base = False  # Crucial for a drone; it needs to fly!
    import_config.make_default_prim = True
    import_config.create_physics_scene = True
    
    # Run the import process
    print(f"Converting {urdf_path} to {dest_usd_path}...")
    _urdf_importer.acquire_urdf_importer_interface().import_urdf(
        urdf_path, 
        dest_usd_path, 
        import_config
    )
    print("Conversion complete!")

if __name__ == "__main__":
    # These paths will be targeted for your cloud directory structure later
    import sys
    if len(sys.argv) > 2:
        convert_urdf_to_usd(sys.argv[1], sys.argv[2])