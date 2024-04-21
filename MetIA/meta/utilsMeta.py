import monai._version


def get_monai_version():
	"""permite to get the monai version which is install on the used computer

	Args:

	Returns:
		return the monai version install on the current computer
	Raises:
			
	See also:
			
	Notes:

	References:
			
	Examples:
	"""
	return tuple(map(lambda val: int(val), monai._version.get_versions()["version"].split(".")))


MONAI_NEWER_THAN_0_8 = get_monai_version() > (0, 8, 0)
