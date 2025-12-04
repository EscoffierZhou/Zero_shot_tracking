import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'CV_TRACKING_ADVANCED'))

print("Testing imports...")
try:
    from CV_TRACKING_ADVANCED.trackers.ostrack_tracker import OSTrackFerrari
    print("✅ Successfully imported OSTrackFerrari")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print("Testing app.py imports...")
try:
    import CV_TRACKING_ADVANCED.app
    print("✅ Successfully imported app")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
