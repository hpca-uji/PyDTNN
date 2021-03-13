PyDTNN development
====================

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
    (develop) $ git checkout new_feature    #  │>  git rebase origin/develop
    (new_feature) $ git rebase develop      #  ┘
    (new_feature) $ git stash pop   # If there were files pending to be committed

To merge the ``new_feature`` branch into ``develop`` (after syncing it with ``develop``
as described in the previous case)::

    (new_feature) $ git checkout develop
    (develop) $ git pull
    (develop) $ git merge new_feature


How to distribute a new version to Pypi
---------------------------------------

Make sure that the package ``checkdocs`` is installed. If not, install it with::

    $ pip3 install --user collective.checkdocs

To create a source distribution::

    $ python3 ./setup.py checkdocs sdist

To test the source distribution ``dist/pydtnn-x.x.x.tar.gz``::

    $ virtualenv --python=python3 testpydtnn
    $ source testpydtnn/bin/activate
    $ pip3 install dist/pydtnn-x.y.z.tar.gz
    $ deactivate

To upload it to the `test pypi repository <https://testpypi.python.org/>`_::

    $ twine upload --repository-url https://test.pypi.org/legacy/ dist/pydtnn-x.x.x.tar.gz

To upload it to the `pypi repository <https://pypi.org/>`_::

    $ twine upload dist/pydtnn-x.x.x.tar.gz


References
----------
[1] https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
