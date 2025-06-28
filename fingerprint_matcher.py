import cv2


# Match two fingerprint images and return result
def match_fingerprints(
    img1_path, img2_path, min_keypoints=20, match_ratio_threshold=0.15
):
    # Read images in grayscale
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return "Error: One or both images could not be loaded."

    # Set up ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Check if there are enough keypoints
    if (
        des1 is None
        or des2 is None
        or len(kp1) < min_keypoints
        or len(kp2) < min_keypoints
    ):
        return "Invalid input: Not a fingerprint or poor quality image."

    # Match descriptors using Brute Force Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [
        m for m in matches if m.distance < 50
    ]  # 50 is a typical threshold for ORB

    # Calculate match ratio
    match_ratio = len(good_matches) / min(len(kp1), len(kp2))
    print(
        f"Keypoints1: {len(kp1)}, Keypoints2: {len(kp2)}, Good matches: {len(good_matches)}, Match ratio: {match_ratio:.2f}"
    )

    if match_ratio > match_ratio_threshold:
        return "Match: Same fingerprint"
    else:
        return "Mismatch: Different fingerprint"


# Run this file to test fingerprint matching
if __name__ == "__main__":
    result = match_fingerprints("finger1.png", "finger2.png")
    print(result)
