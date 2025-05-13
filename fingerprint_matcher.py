
import cv2

def match_fingerprints(img1_path, img2_path, threshold=10):
    # Load grayscale images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return "Error: One or both images could not be loaded."

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return "Error: Could not compute descriptors for one or both images."

    # Match descriptors using Brute Force Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)
    match_score = len(matches)

    print(f"Match score: {match_score}")
    return "Match: Same fingerprint" if match_score > threshold else "Mismatch: Different fingerprint"

# Example test run
if __name__ == "__main__":
    result = match_fingerprints("finger1.png", "finger2.png")
    print(result)
