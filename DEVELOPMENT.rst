PyDTNN development
====================

How to modify PyDTNN and execute the modified version
-----------------------------------------------------

In order to be able to run ``pydtnn_benchmark`` or import ``pydtnn``, the PyDTNN
module must be installed first. Although it is possible to install PyDTNN from
the project path (i.e., there is no need to build a source distribution file
first), this option implies that the PyDTNN code will be installed on the
``site-packages/pydtnn`` folder, which is not convenient if you plan to edit its
code.


Fortunately, it is possible to instruct the ``pip`` command to install a project
**in editable mode** from a local project path. Therefore, it suffices to execute
the next command on the project path to be able to edit the code and then
execute its edited versions::

    $ pip install -e .

The previous command must be run at least once. It is not necessary to execute
it each time that a newer version is going to be tested. Unless the cython
modules have been modified and, therefore, they should be recompiled!


How to collaborate
------------------

The git workflow to be used is the gitflow-workflow [1]. The new features and
fixes should be incorporated first in the ``develop`` branch from the private
PyDTNN repository. All the work must be done on branches derived from the
``develop`` branch.

To create a ``new_feature`` branch from develop::

    (develop) $ git pull
    (develop) $ git branch new_feature
    (develop) $ git checkout new_feature
    (new_feature) $

To upload the ``new_feature`` branch to the repository (only if the new feature
is going to be developed collaboratively)::

    (new_feature) $ git push -u origin new_feature

If there are changes on the ``develop`` branch that should be synced with the
``new_feature`` branch::

    (new_feature) $ git stash       # If there are files pending to be committed
    (new_feature) $ git checkout develop    #  ┐
    (develop) $ git pull                    #  │
    (develop) $ git checkout new_feature    #  │>  git rebase origin/develop ?
    (new_feature) $ git rebase develop      #  ┘
    (new_feature) $ git stash pop   # If there were files pending to be committed

To merge the ``new_feature`` branch into ``develop`` (after syncing it with ``develop``
as described in the previous case)::

    (new_feature) $ git checkout develop
    (develop) $ git pull
    (develop) $ git merge new_feature


How to interactively debug the code
-----------------------------------

Among the many options to do this, one is to use the ``ipdb`` module, which
allows to enclose the part of the code that fails with::

    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        [...]

Using the previous construction it is also possible to force a debugging session
by manually issuing an exception inside the with block.


How to upload a new version to Pypi
-----------------------------------

Make sure that the package ``checkdocs`` is installed. If not, install it with::

    $ pip install --user collective.checkdocs

To create a source distribution::

    $ python ./setup.py checkdocs sdist

To test the source distribution ``dist/pydtnn-x.x.x.tar.gz``::

    $ virtualenv --python=python3 testpydtnn
    $ source testpydtnn/bin/activate
    $ pip install dist/pydtnn-x.y.z.tar.gz
    $ deactivate

To upload it to the `test pypi repository <https://testpypi.python.org/>`_::

    $ twine upload --repository-url https://test.pypi.org/legacy/ dist/pydtnn-x.x.x.tar.gz

To upload it to the `pypi repository <https://pypi.org/>`_::

    $ twine upload dist/pydtnn-x.x.x.tar.gz


References
----------
[1] https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
