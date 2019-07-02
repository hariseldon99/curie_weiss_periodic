from distutils.core import setup, Extension
import numpy as np

setup (name = 'curie_weiss_periodic',
        version = '0.1',
        description = """Exact Dynamics of a Curie-Weiss model of spins""",
        long_description=\
      """
        Exact Dynamics for quantum
        spins and transverse fields with time-periodic drive

        * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        * Copyright (c) 2018 Analabha Roy (daneel@utexas.edu)
        *
        *This is free software: you can redistribute it and/or modify
        *it under the terms of version 2 of the GNU Lesser General
        *Public License as published by the Free Software Foundation.
        *Notes:
        *1. The initial state is currently hard coded to be the
        *classical ground  state of the Transverse Field Ising model
        * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         """,
        url='https://github.com/hariseldon99/curie_weiss_periodic',
         # Author details
    author='Analabha Roy',
    author_email='daneel@utexas.edu',
    package_data={'': ['LICENSE']},
        include_package_data=True,

    # Choose your license
    license='GPL',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Physicists',
        'Topic :: Numerical Quantum Simulations :: Dynamics',

        # Pick your license as you wish (should match "license" above)
        'License :: GPL License',

        # Specify the Python versions you support here. In particular,
        #ensure that you indicate whether you support Python 2,
        # Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
        packages=['curie_weiss_periodic'],
        )
