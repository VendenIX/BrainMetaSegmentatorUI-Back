def is_notebook() -> bool:
	"""We want to verify if the used code is a notebook.
	first, import a package use to verify if the current shell is a notebook.
	second, use the package to have the current interactive shelle, if it's none, return False
	third, put in the variable "shell" the name of the class return by get_ipython()
	fourth, if "shell" or get_ipython().__class__.__module as the name of a interactive shell in a notebook, return True
			else, if "shell" is name of a interactive name in a notebook, return false
			else, in all of the other case, return false

	Args:

	Returns:
		a boolean which say if we are in a notebook or not
	Raises:
		if we have an error in this function, return false, it's probably saying that our current interactive shell is a standard pytho interpreter
	See also:
			
	Notes:

	References:
			
	Examples:
	"""
	try:
		from IPython import get_ipython

		if get_ipython() is None:
			return False
		
		shell = get_ipython().__class__.__name__
		if shell == 'ZMQInteractiveShell' or get_ipython().__class__.__module__ == "google.colab._shell":
			return True   # Jupyter notebook or qtconsole or Google Colab
		elif shell == 'TerminalInteractiveShell':
			return False  # Terminal running IPython
		else:
			return False  # Other type (?)
	except NameError:
		return False      # Probably standard Python interpreter
