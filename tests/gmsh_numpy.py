#!/usr/bin/python2.7 -u
# -*- coding: utf-8 -*-\

'''
    IO functions for Gmsh .msh files
    This program is part of the SimNIBS package.
    Please check on www.simnibs.org how to cite our work in publications.


    Copyright (C) 2015 Andre Antunes, Guilherme B Saturnino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.



This file Incorpotates work covered by the following copyright and permission notice

    Copyright (c) 2005-2013, NumPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

import os
import struct
import copy

import numpy as np
import scipy.spatial
import h5py


# =============================================================================
# CLASSES
# =============================================================================

class Nodes:
    """class to handle the node information:

    Parameters:
    -----------------------
    node_coord (optional): (Nx3) ndarray
        Coordinates of the nodes

    Attributes:
    ----------------------
    node_coord: (Nx3) ndarray
        Coordinates of the nodes

    node_number: (Nx1) ndarray
        ID of nodes

    units:
        Name of units

    nr: property
        Number of nodes

    Examples
    -----------------------------------
     >>> nodes.node_coord
     array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
     >>> nodes.node_number
     array([1, 2, 3])
     >>> nodes[1]
     array([1, 0, 0])
     >>> nodes[array(True, False, True)]
     array([[1, 0, 0], [0, 0, 1]])
    """

    def __init__(self, node_coord=None):
        # gmsh fields
        self.node_number = np.array([], dtype='int32')
        self.node_coord = np.array([], dtype='float64')

        # simnibs fields
        self.units = 'mm'

        if node_coord is not None:
            self.set_nodes(node_coord)

    @property
    def nr(self):
        return self.node_coord.shape[0]

    def set_nodes(self, node_coord):
        """Sets the nodes

        Parameters
        ----------------------
        node_coords: Nx3 array
            array of node positions
        """

        assert node_coord.shape[1] == 3
        self.node_coord = node_coord.astype('float64')
        self.node_number = np.array(list(range(1, self.nr + 1)), dtype='int32')

    def find_closest_node(self, querry_points, return_index=False):
        """ Finds the closest node to each point in p

        Parameters
        --------------------------------
        querry_points: (Nx3) ndarray
            List of points (x,y,z) to which the closes node in the mesh should be found

        return_index: (optional) bool
            Whether to return the index of the closes nodes

        Returns
        -------------------------------
        coords: Nx3 array of floats
            coordinates of the closest points

        indexes: Nx1 array of ints
            Indices of the nodes in the mesh

        --------------------------------------
        The indices are in the mesh listing, that starts at one!
       """
        if len(self.node_coord) == 0:
            raise ValueError('Mesh has no nodes defined')

        kd_tree = scipy.spatial.cKDTree(self.node_coord)
        _, indexes = kd_tree.query(querry_points)
        coords = self.node_coord[indexes, :]

        if return_index:
            return (coords, self.node_number[indexes])

        else:
            return coords

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__
        except AttributeError:
            return False

    def __getitem__(self, index):
        rel = -99999 * np.ones(max(self.node_number) + 1, dtype=int)
        rel[self.node_number] = np.arange(0, len(self.node_number), dtype=int)
        node_coord_index = rel[index]
        if np.any(node_coord_index == -99999):
            raise IndexError('Invalid node number')
        return self.node_coord[node_coord_index]


class Elements:
    """class to handle the element information. Contains:

    Can only handle triangles and tetrahedra!

    Parameters
    --------------------------
    triangles (optional): (Nx3) ndarray
        List of nodes composing each triangle
    tetrahedra(optional): (Nx3) ndarray
        List of nodes composing each tetrahedra


    Attributes
    ----------------------------------
    elm_number: (Nx1) ndarray
          element ID (usuallly from 1 till number_of_elements)
    elm_type: (Nx1) ndarray
        elm-type (2=triangle, 4=tetrahedron, etc)
    tag1: (Nx1) ndarray
        first tag for each element
    tag2: (Nx1) ndarray
        second tag for each elenent
    node_number_list: (Nx4) ndarray
        4xnumber_of_element matrix of the nodes that constitute the element.
        For the triangles, the fourth element = 0


    Notes
    -------------------------
    Node and element count starts at 1!
    """

    def __init__(self, triangles=None, tetrahedra=None):
        # gmsh fields
        self.elm_number = np.array([], 'int32')
        self.elm_type = np.array([], 'int8')
        self.tag1 = np.array([], dtype='int16')
        self.tag2 = np.array([], dtype='int16')
        self.node_number_list = np.array([], dtype='int32')

        if triangles is not None:
            assert triangles.shape[1] == 3
            assert np.all(triangles > 0), "Node count should start at 0"
            self.node_number_list = np.zeros(
                (triangles.shape[0], 4), dtype='int32')
            self.node_number_list[:, :3] = triangles.astype('int32')
            self.elm_type = np.ones((self.nr,), dtype='int32') * 2

        if tetrahedra is not None:
            assert tetrahedra.shape[1] == 4
            if len(self.node_number_list) == 0:
                self.node_number_list = tetrahedra.astype('int32')
                self.elm_type = np.ones((self.nr,), dtype='int32') * 4
            else:
                self.node_number_list = np.vstack(
                    (self.node_number_list, tetrahedra.astype('int32')))
                self.elm_type = np.append(
                    self.elm_type, np.ones((self.nr,), dtype='int32') * 4)

        if len(self.node_number_list) > 0:
            self.tag1 = np.ones((self.nr,), dtype='int32')
            self.tag2 = np.ones((self.nr,), dtype='int32')
            self.elm_number = np.array(list(range(1, self.nr + 1)), dtype='int32')

    @property
    def nr(self):
        return self.node_number_list.shape[0]

    @property
    def triangles(self):
        return self.elm_number[self.elm_type == 2]

    @property
    def tetrahedra(self):
        return self.elm_number[self.elm_type == 4]

    def Print(self):
        print("Elements info:")
        print("nr elements:  ", self.nr)
        print("nodes of last elm", self.node_number_list[-1])

    def find_all_elements_with_node(self, node_nr):
        """ Finds all elements that have a given node

        Parameters
        -----------------
        node_nr: int
            number of node

        Returns
        ---------------
        elm_nr: np.ndarray
            array with indices of element numbers

        """
        elm_with_node = np.array(
            sum(self.node_number_list[:, i] == node_nr for i in range(4)),
            dtype=bool)
        return self.elm_number[elm_with_node]

    def find_neighbouring_nodes(self, node_nr):
        """ Finds the nodes that share an element with the specified node

        Parameters
        -----------------------------
        node_nr: int
            number of query node (mesh listing)
        Returns
        ------------------------------
        all_neighbours: np.ndarray
            list of all nodes what share an element with the node

        Example
        ----------------------------
        >>> elm = gmsh.Elements()
        >>> elm.node_number_list = np.array([[1, 2, 3, 4], [3, 2, 5, 7], [1, 8, 2, 6]])
        >>> elm.find_neighbouring_nodes(1))
        array([2, 3, 4, 8, 6])

        """
        elm_with_node = np.array(
            sum(self.node_number_list[:, i] == node_nr for i in range(4)),
            dtype=bool)
        all_neighbours = self.node_number_list[elm_with_node, :].reshape(-1)
        all_neighbours = np.unique(all_neighbours)
        all_neighbours = all_neighbours[all_neighbours != node_nr]
        all_neighbours = all_neighbours[all_neighbours != 0]
        return np.array(all_neighbours, dtype=int)

    def __getitem__(self, index):
        rel = -99999 * np.ones(max(self.elm_number) + 1, dtype=int)
        rel[self.elm_number] = np.arange(0, self.nr, dtype=int)
        node_number_list_index = rel[index]
        if np.any(node_number_list_index == -99999):
            raise IndexError('Invalid element number')
        return self.node_number_list[node_number_list_index]

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__
        except AttributeError:
            return False


class Msh:
    """class to handle the meshes.
    Gatters Nodes, Elements and Data

    Parameters:
    -------------------------
    Nodes: simnibs.msh.gmsh_numpy.Nodes
        Nodes structures

    Elements: simnibs.msh.gmsh_numpy.Elements()
        Elements structure

    Attributes:
    -------------------------
    nodes: simnibs.msh.gmsh_numpy.Nodes
        a Nodes field
    elm: simnibs.msh.gmsh_numpy.Elements
        A Elements field
    nodedata: simnibs.msh.gmsh_numpy.NodeData
        list of NodeData filds
    elmdata: simnibs.msh.gmsh_numpy.ElementData
        list of ElementData fields
    fn: str
        name of file
    binary: bool
        wheather or not the mesh was in binary format
   """

    def __init__(self, nodes=None, elements=None):
        self.nodes = Nodes()
        self.elm = Elements()
        self.nodedata = []
        self.elmdata = []
        self.fn = ''  # file name to save msh
        self.binary = False

        if nodes is not None:
            self.nodes = nodes
        if elements is not None:
            self.elements = elements

    @property
    def field(self):
        return dict(
            [(data.field_name, data) for data in self.elmdata + self.nodedata])

    def crop_mesh(self, tags=None, elm_type=None):
        """ Crops the specified tags from the mesh
        Generates a new mesh, with only the specified tags
        The nodes are also reordered

        Parameters:
        ---------------------
        tags:(optinal) int or list
            list of tags to be cropped, default: all

        elm_type: (optional) list of int
            list of element types to be croped, default: all

        Returns:
        ---------------------
        simnibs.msh.gmsh_numpy.Msh
            Mesh with only the specified tags

        Raises:
        -----------------------
            ValueError, if the tag and elm_type combination is not foud
        """
        if tags is None and elm_type is None:
            raise ValueError("Either a list of tags or element types must be specified")

        elm_remove = np.ones((self.elm.nr, ), dtype=bool)

        if tags is not None:
            elm_remove *= np.logical_not(np.in1d(self.elm.tag1, tags))

        if np.all(elm_remove) is True and tags is not None:
            raise ValueError("Tags: ", tags, "not found in mesh")

        if elm_type is not None:
            elm_remove *= np.logical_not(np.in1d(self.elm.elm_type, elm_type))

        if np.all(elm_remove) is True and elm_type is not None:
            raise ValueError("Tags: {0} and Element: {1}"
                             "Combination not found in mesh".format(tags, elm_type))

        idx = np.where(np.logical_not(elm_remove))[0]
        nr_elements = len(idx)
        unique_nodes = np.unique(self.elm.node_number_list[idx, :].reshape(-1))
        if unique_nodes[0] == 0:
            unique_nodes = np.delete(unique_nodes, 0)
        nr_unique = np.size(unique_nodes)

        # creates a dictionary
        nodes_dict = np.zeros(self.nodes.nr + 1, dtype='int')
        nodes_dict[unique_nodes] = np.array(
            list(range(1, 1 + nr_unique)), dtype='int32')

        # Gets the new node numbers
        node_number_list = nodes_dict[self.elm.node_number_list[idx, :]]

        # and the positions in appropriate order
        node_coord = self.nodes.node_coord[unique_nodes - 1]

        # gerenates new mesh
        cropped = Msh()

        cropped.elm.elm_number = np.array(
            list(range(1, 1 + nr_elements)), dtype='int32')
        cropped.elm.tag1 = np.copy(self.elm.tag1[idx])
        cropped.elm.tag2 = np.copy(self.elm.tag2[idx])
        cropped.elm.elm_type = np.copy(self.elm.elm_type[idx])
        cropped.elm.node_number_list = np.copy(node_number_list)

        cropped.nodes.node_number = np.array(
            list(range(1, 1 + nr_unique)), dtype='int32')
        cropped.nodes.node_coord = np.copy(node_coord)

        cropped.nodedata = copy.deepcopy(self.nodedata)

        for nd in cropped.nodedata:
            nd.node_number = np.array(list(range(1, 1 + nr_unique)), dtype='int32')
            if nd.nr_comp == 1:
                nd.value = np.copy(nd.value[unique_nodes - 1])
            else:
                nd.value = np.copy(nd.value[unique_nodes - 1, :])

        cropped.elmdata
        for ed in self.elmdata:
            cropped.elmdata.append(ElementData(ed.value[idx], ed.field_name))

        return cropped

    def elements_baricenters(self, elements='all'):
        """ Calculates the baricenter of the elements
        Parameters
        ------------
        elements(optional): np.ndarray
            elements where the baricenters should be calculated. default: all elements

        Returns
        -----------------------------------
        baricenters: ElementData
            ElementData with the baricentes of the elements
        """
        if elements is 'all':
            elements = self.elm.elm_number
        bar = ElementData(np.zeros((len(elements), 3), dtype=float),
                          'baricenter')
        bar.elm_number = elements
        th_indexes = np.intersect1d(self.elm.tetrahedra, elements)
        tr_indexes = np.intersect1d(self.elm.triangles, elements)

        if len(th_indexes) > 0:
            bar[th_indexes] = np.average(
                self.nodes[self.elm[th_indexes]],
                axis=1)

        if len(tr_indexes) > 0:
            bar[tr_indexes] = np.average(
                self.nodes[self.elm[tr_indexes][:, :3]],
                axis=1)

        return bar

    def elements_volumes_and_areas(self, elements='all'):
        """ Calculates the volumes of tetrahedra and areas of triangles

        Parameters
        --------------
        elements: np.ndarray
            elements where the volumes / areas should be calculated
            default: all elements

        Retuns
        ---------------------------------------
        ndarray
            Volume/areas of tetrahedra/triangles

        Note
        --------------------------------------
        In the mesh's unit (normally mm)
        """
        if elements is 'all':
            elements = self.elm.elm_number
        vol = ElementData(np.zeros((len(elements),), dtype=float),
                          'volumes_and_areas')
        vol.elm_number = elements
        th_indexes = np.intersect1d(self.elm.tetrahedra, elements)
        tr_indexes = np.intersect1d(self.elm.triangles, elements)

        sideA = self.nodes[self.elm[tr_indexes][:, 1]] - \
            self.nodes[self.elm[tr_indexes][:, 0]]

        sideB = self.nodes[self.elm[tr_indexes][:, 2]] - \
            self.nodes[self.elm[tr_indexes][:, 0]]

        n = np.cross(sideA, sideB)
        vol[tr_indexes] = np.linalg.norm(n, axis=1) * 0.5

        sideA = self.nodes[self.elm[th_indexes][:, 1]] - \
            self.nodes[self.elm[th_indexes][:, 0]]

        sideB = self.nodes[self.elm[th_indexes][:, 2]] - \
            self.nodes[self.elm[th_indexes][:, 0]]

        sideC = self.nodes[self.elm[th_indexes][:, 3]] - \
            self.nodes[self.elm[th_indexes][:, 0]]

        vol[th_indexes] = 1. / 6. * \
            np.abs(np.sum(sideC * np.cross(sideA, sideB), axis=1))

        return vol

    def find_closest_element(self, querry_points, return_index=False):
        """ Finds the closest node to each point in p

        Parameters
        --------------------------------
        querry_points: (Nx3) ndarray
            List of points (x,y,z) to which the closes node in the mesh should be found

        return_index: (optional) bool
            Whether to return the index of the closes nodes, default=False

        Returns
        -------------------------------
        coords: Nx3 array of floats
            coordinates of the baricenter of the closest element

        indexes: Nx1 array of ints
            Indice of the closest elements

        Notes
        --------------------------------------
        The indices are in the mesh listing, that starts at one!

        """
        if len(self.elm.node_number_list) == 0:
            raise ValueError('Mesh has no elements defined')

        baricenters = self.elements_baricenters()
        kd_tree = scipy.spatial.cKDTree(baricenters.value)
        _, indexes = kd_tree.query(querry_points)
        indexes = baricenters.elm_number[indexes]
        coords = baricenters[indexes]

        if return_index:
            return (coords, indexes)

        else:
            return coords

    def elm_node_coords(self, elm_nr=None, tag=None, elm_type=None):
        """ Returns the position of each of the element's nodes

        Arguments
        -----------------------------
        elm_nr: (optional) array of ints
            Elements to return, default: Return all elements
        tag: (optional) array of ints
            Only return elements with specified tag. default: all tags
        elm_type: (optional) array of ints
            Only return elements of specified type. default: all

        Returns
        -----------------------------
        Nx4x3 ndarray
            Array with node position of every element
            For triangles, the fourth coordinates are 0,0,0
        """
        elements_to_return = np.ones((self.elm.nr, ), dtype=bool)

        if elm_nr is not None:
            elements_to_return[elm_nr] = True

        if elm_type is not None:
            elements_to_return = np.logical_and(
                elements_to_return,
                np.in1d(self.elm.elm_type, elm_type))

        if tag is not None:
            elements_to_return = np.logical_and(
                elements_to_return,
                np.in1d(self.elm.tag1, tag))

        tmp_node_coord = np.vstack((self.nodes.node_coord, [0, 0, 0]))

        elm_node_coords = \
            tmp_node_coord[self.elm.node_number_list[elements_to_return, :] - 1]

        return elm_node_coords

    def write_hdf5(self, hdf5_fn, path='./'):
        """ Writes a HDF5 file with mesh information

        Parameters
        -----------
        hdf5_fn: str
            file name of hdf5 file
        path: str
            path in the hdf5 file where the mesh should be saved
        """
        with h5py.File(hdf5_fn, 'a') as f:
            try:
                g = f.create_group(path)
            except ValueError:
                g = f[path]
            if 'elm' in list(g.keys()):
                raise IOError('Cannot write mesh in group: {0} '
                              'in file: {1}, this group has a mesh '
                              'already'.format(path, hdf5_fn))
            g.attrs['fn'] = self.fn
            elm = g.create_group('elm')
            for key, value in vars(self.elm).items():
                elm.create_dataset(key, data=value)
            node = g.create_group('nodes')
            for key, value in vars(self.nodes).items():
                node.create_dataset(key, data=value)
            fields = g.create_group('fields')
            for key, value in self.field.items():
                field = fields.create_group(key)
                field.attrs['type'] = value.__class__.__name__
                field.create_dataset('value', data=value.value)

    def read_hdf5(self, hdf5_fn, path='./'):
        """ Reads mesh information from an hdf5 file

        Parameters
        ----------
        hdf5_fn: str
            file name of hdf5 file
        path: str
            path in the hdf5 file where the mesh is saved
        """
        with h5py.File(hdf5_fn, 'r') as f:
            g = f[path]
            self.fn = g.attrs['fn']
            for key, value in self.elm.__dict__.items():
                setattr(self.elm, key, np.array(g['elm'][key]))
            for key, value in self.nodes.__dict__.items():
                setattr(self.nodes, key, np.array(g['nodes'][key]))
            self.nodes.units = str(self.nodes.units)
            for field_name, field_group in g['fields'].items():
                if field_group.attrs['type'] == 'ElementData':
                    self.elmdata.append(ElementData(np.array(field_group['value']),
                                                    field_name))
                if field_group.attrs['type'] == 'NodeData':
                    self.nodedata.append(NodeData(np.array(field_group['value']),
                                                  field_name))

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__
        except AttributeError:
            return False

    def tetrahedra_quality(self, tetrahedra='all'):
        """ calculates the quality measures of the tetrahedra

        Parameters
        ------------
        tetrahedra: np.ndarray
            tags of the tetrahedra where the quality parameters are to be calculated

        Returns
        ----------
        measures: dict
            dictionary with ElementData with measures
        """
        if tetrahedra is 'all':
            tetrahedra = self.elm.tetrahedra
        if not np.all(np.in1d(tetrahedra, self.elm.tetrahedra)):
            raise ValueError('No tetrahedra with element number'
                             '{0}'.format(tetrahedra[
                                 np.logical_not(
                                     np.in1d(tetrahedra, self.elm.tetrahedra))]))
        M = self.nodes[self.elm[tetrahedra]]
        measures = {}
        V = self.elements_volumes_and_areas(tetrahedra)[tetrahedra]
        E = np.array([
            M[:, 0] - M[:, 1],
            M[:, 0] - M[:, 2],
            M[:, 0] - M[:, 3],
            M[:, 1] - M[:, 2],
            M[:, 1] - M[:, 3],
            M[:, 2] - M[:, 3]])
        E = np.swapaxes(E, 0, 1)
        S = np.linalg.norm(E, axis=2)
        # calculate the circunstribed radius
        a = S[:, 0] * S[:, 5]
        b = S[:, 1] * S[:, 4]
        c = S[:, 2] * S[:, 3]
        s = 0.5 * (a + b + c)
        delta = np.sqrt(s * (s - a) * (s - b) * (s - c))
        CR = delta / (6 * V)
        # radius or inscribed sphere
        SA = np.linalg.norm([
            np.cross(E[:, 0], E[:, 1]),
            np.cross(E[:, 0], E[:, 2]),
            np.cross(E[:, 1], E[:, 2]),
            np.cross(E[:, 3], E[:, 4])], axis=2) * 0.5
        SA = np.swapaxes(SA, 0, 1)
        IR = 3 * V / np.sum(SA, axis=1)
        measures['beta'] = ElementData(CR / IR, 'beta')
        measures['beta'].elm_number = tetrahedra

        gamma = (np.sum(S * S, axis=1) / 6.) ** (3./2.) / V
        measures['gamma'] = ElementData(gamma, 'gamma')
        measures['gamma'].elm_number = tetrahedra
        return measures


class Data(object):
    """Store data in elements or nodes

    Parameters
    -----------------------
    value: np.ndarray
        Value of field in nodes

    field_name: str
        name of field

    Attributes
    --------------------------
    value: ndarray
        Value of field in nodes
    field_name: str
        name of field
    nr: property
        number of data points
    nr_comp: property
        number of dimensions per data point (1 for scalars, 3 for vectors)

    """

    def __init__(self, value=np.empty(0, dtype=float), name=''):
        self.field_name = name
        self.value = value

    @property
    def type(self):
        return self.__class__.__name__

    @property
    def nr(self):
        return self.value.shape[0]

    @property
    def nr_comp(self):
        try:
            return self.value.shape[1]
        except IndexError:
            return 1

    @property
    def indexing_nr(self):
        raise Exception('indexing_nr is not defined')

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__
        except AttributeError:
            return False

    def __getitem__(self, index):
        rel = -99999 * np.ones(max(self.indexing_nr) + 1, dtype=int)
        rel[self.indexing_nr] = np.arange(0, len(self.indexing_nr), dtype=int)
        value_index = rel[index]
        if np.any(value_index == -99999):
            raise IndexError('Invalid index')
        return self.value[value_index]

    def __setitem__(self, index, item):
        rel = -99999 * np.ones(max(self.indexing_nr) + 1, dtype=int)
        rel[self.indexing_nr] = np.arange(0, len(self.indexing_nr), dtype=int)
        value_index = rel[index]
        if np.any(value_index == -99999):
            raise IndexError('Invalid index')
        self.value[value_index] = item

    def add_to_hdf5_leadfield(self, leadfield_fn, row, path=None, order='C', nbr_rows=None):
        """ Adds the field as a row in a hdf5 leadfield

        Parameters
        -----------
        leadfield_fn: str
            name of hdf5 file
        row: int
            row where the leadfield should be added
        path: str
            Path in the hdf5 file. Default:'leadfield/field_name'
        order: 'C' of 'F'
            order in which multidimensional data should be flattened
        nbr_rows: int
            total number of rows in the leadfield
        """
        if path is None:
            path = 'leadfield/' + self.field_name
        with h5py.File(leadfield_fn, 'a') as f:
            try:
                f[path]
            except KeyError:
                if nbr_rows is None:
                    shape = (1, self.nr*self.nr_comp)
                else:
                    shape = (nbr_rows, self.nr*self.nr_comp)
                f.create_dataset(path, shape=shape,
                                 maxshape=(nbr_rows, self.nr*self.nr_comp),
                                 dtype='float64')

            if row >= f[path].shape[0]:
                f[path].resize(row+1, axis=0)
            f[path][row, :] = self.value.flatten(order=order)

    @classmethod
    def read_hdf5_leadfield_row(cls, leadfield_fn, field_name, row, nr_components,
                                path=None, order='C'):
        """ Reads a row of an hdf5 leadfield and store it as Data

        Parameters
        -----------
        leadfield_fn: str
            Name of file with leadfield
        field_name: str
            name of field
        row: int
            number of the row to be read
        nr_components: int
            number of components of the field (1 for scalars, 3 for vectors, 9 for
            tensors)
        path (optional): str
            Path to the leadfiend in the hdf5. Default: leadfiels/field_name
        order: 'C' or 'F'
            in which order the multiple components of the field are stored

        Returns
        ---------
        data: nnav.Data()
            instance with the fields
        """
        if path is None:
            path = 'leadfield/' + field_name
        with h5py.File(leadfield_fn, 'r') as f:
            value = np.reshape(f[path][row, :], (-1, nr_components), order=order)
        return cls(value, field_name)

class ElementData(Data):
    """Store data in elements

    Parameters
    -----------------------
    value: ndarray
        Value of field in nodes

    field_name: str
        name of field

    Attributes
    --------------------------
    value: ndarray
        Value of field in elements
    field_name: str
        name of field
    elm_number: ndarray
        index of elements
    nr: property
        number of data points
    nr_comp: property
        number of dimensions per data point (1 for scalars, 3 for vectors)
    """

    def __init__(self, value=[], name=''):
        Data.__init__(self, value=value, name=name)
        self.elm_number = np.array([], dtype='int32')

        if len(value) > 0:
            self.elm_number = np.array(list(range(1, self.nr + 1)), dtype='int32')

    @property
    def indexing_nr(self):
        return self.elm_number

    def elm_data2node_data(self, msh, k=(10, 6)):
        """Transforms an ElementData field into a NodeData field
        Uses cKDTree to perform the interpolation.

        Parameters:
        ----------------------
        msh: simnibs.gmsh_numpy.Msh
            mesh structure with the geometrical information
        k: (int,int), optional
            Number of nearest neighbours to be taken into account for tetrahedra
            and triangles respectivelly

        Returns:
        ----------------------
        simnibs.gmsh_numpy.NodeData
            Structure with NodeData

        """
        if self.nr != msh.elm.nr:
            raise ValueError("The number of data points in the data "
                             "structure should be equal to the number of elements in the mesh")

        node_data = np.zeros((msh.nodes.nr, self.nr_comp), dtype=float)

        vol_tags = np.unique(msh.elm.tag1[msh.elm.elm_type == 4]).tolist()
        for vt in vol_tags:
            th_indexes = np.where(np.logical_and(
                msh.elm.elm_type == 4, msh.elm.tag1 == vt))[0]
            th_baricenters = np.average(
                msh.nodes.node_coord[msh.elm.node_number_list[th_indexes, :4] - 1], axis=1)
            kd = scipy.spatial.cKDTree(th_baricenters)

            nodes = np.unique(msh.elm.node_number_list[th_indexes, :])
            dists, tetra = kd.query(msh.nodes.node_coord[nodes - 1, :], k[0])

            if k[0] == 1:
                dists = dists[:, np.newaxis]
                tetra = tetra[:, np.newaxis]

            if self.nr_comp == 1:
                field_value = self.value[th_indexes].reshape(-1)
                node_data[nodes - 1] = \
                    (np.sum(field_value[tetra] * (dists**(-2)), axis=1) /
                     np.sum(dists**(-2), axis=1))[:, np.newaxis]
            else:
                field_value = self.value[th_indexes]
                node_data[nodes - 1, :] = np.sum(field_value[tetra] *
                                                 dists[:, :, np.newaxis]**(-2),
                                                 axis=1) /\
                    (np.sum(dists**(-2), axis=1)[:, np.newaxis])

        surface_tags = np.unique(msh.elm.tag1[msh.elm.elm_type == 2]).tolist()
        for st in surface_tags:
            tr_indexes = np.where(np.logical_and(
                msh.elm.elm_type == 2, msh.elm.tag1 == st))[0]

            tr_baricenters = np.average(
                msh.nodes.node_coord[msh.elm.node_number_list[tr_indexes, :3] - 1], axis=1)
            kd = scipy.spatial.cKDTree(tr_baricenters)

            nodes = np.unique(msh.elm.node_number_list[tr_indexes, :3])
            dists, tr = kd.query(msh.nodes.node_coord[nodes - 1, :], k[1])

            if k[1] == 1:
                dists = dists[:, np.newaxis]
                tr = tr[:, np.newaxis]

            if self.nr_comp == 1:
                field_value = self.value[tr_indexes].reshape(-1)
                node_data[nodes - 1] = (np.sum(field_value[tr] * (dists**(-2)), axis=1) /
                                        (np.sum(dists**(-2), axis=1)))[:, np.newaxis]
            else:
                field_value = self.value[tr_indexes]
                node_data[nodes - 1, :] = np.sum(field_value[tr] *
                                                 dists[:, :, np.newaxis]**(-2), axis=1) /\
                    (np.sum(dists**(-2), axis=1)[:, np.newaxis])

        return NodeData(node_data, self.field_name)

    def append_to_mesh(self, fn, mode='binary'):
        """Appends this ElementData fields to a file

        Parameters:
        --------------------------
            fn: str
                file name
            mode: binary or ascii
                mode in which to write
        """
        with open(fn, 'a') as f:
            f.write('$ElementData' + '\n')
            # string tags
            f.write(str(1) + '\n')
            f.write('"' + self.field_name + '"\n')

            f.write(str(1) + '\n')
            f.write(str(0) + '\n')

            f.write(str(4) + '\n')
            f.write(str(0) + '\n')
            f.write(str(self.nr_comp) + '\n')
            f.write(str(self.nr) + '\n')
            f.write(str(0) + '\n')

            if mode == 'ascii':
                for ii in range(self.nr):
                    f.write(str(self.elm_number[ii]) + ' ' +
                            str(self.value[ii]).translate(None, '[](),') +
                            '\n')

            elif mode == 'binary':

                elm_number = self.elm_number.astype('int32')
                value = self.value.astype('float64')
                try:
                    value.shape[1]
                except IndexError:
                    value = value[:, np.newaxis]
                m = elm_number[:, np.newaxis]
                for i in range(self.nr_comp):
                    m = np.concatenate((m,
                                        value[:, i].astype('float64').view('int32').reshape(-1, 2)),
                                       axis=1)
                f.write(m.tostring())

            else:
                raise IOError("invalid mode:", mode)

            f.write('$EndElementData\n')


class NodeData(Data):
    """
    Parameters
    -----------------------
    value: ndarray
        Value of field in nodes

    field_name: str
        name of field

    Attributes
    --------------------------
    value: ndarray
        Value of field in elements
    field_name: str
        name of field
    node_number: ndarray
        index of elements
    nr: property
        number of data points
    nr_comp: property
        number of dimensions per data point (1 for scalars, 3 for vectors)
    """

    def __init__(self, value=[], name=''):
        Data.__init__(self, value=value, name=name)
        self.node_number = np.array([], dtype='int32')
        if len(value) > 0:
            self.node_number = np.array(list(range(1, self.nr + 1)), dtype='int32')

    @property
    def indexing_nr(self):
        return self.node_number

    def node_data2elm_data(self, msh):
        """Transforms an ElementData field into a NodeData field
        the value in the element is the average of the value in the nodes

        Parameters:
        ----------------------
        msh: simnibs.gmsh_numpy.Msh
            mesh structure with the geometrical information

        Returns:
        ----------------------
        simnibs.gmsh_numpy.ElementData
            structure with field value interpolated at element centers

        """
        if (self.nr != msh.nodes.nr):
            raise ValueError(
                "The number of data points in the data structure should be"
                "equal to the number of elements in the mesh")

        triangles = np.where(msh.elm.elm_type == 2)[0]
        tetrahedra = np.where(msh.elm.elm_type == 4)[0]

        if self.nr_comp == 1:
            elm_data = np.zeros((msh.elm.nr,), dtype=float)
        else:
            elm_data = np.zeros((msh.elm.nr, self.nr_comp), dtype=float)

        if len(triangles) > 0:
            elm_data[triangles] = \
                np.average(self.value[msh.elm.node_number_list[
                           triangles, :3] - 1], axis=1)
        if len(tetrahedra) > 0:
            elm_data[tetrahedra] = \
                np.average(self.value[msh.elm.node_number_list[
                           tetrahedra, :4] - 1], axis=1)

        return ElementData(elm_data, self.field_name)

    def gradient(self, mesh):
        if self.nr_comp != 1:
            raise ValueError('can only take gradient of scalar fields')
        if mesh.nodes.nr != self.nr:
            raise ValueError('mesh must have the same number of nodes as the NodeData')

        elm_node_coords = mesh.nodes[mesh.elm[mesh.elm.tetrahedra]]

        tetra_matrices = elm_node_coords[:, 1:4, :] - \
            elm_node_coords[:, 0, :][:, None]

        inv_tetra_matrices = np.linalg.inv(tetra_matrices)

        dif_between_tetra_nodes = self[mesh.elm[mesh.elm.tetrahedra][:, 1:4]] - \
            self[mesh.elm[mesh.elm.tetrahedra][:, 0]][:, None]

        th_grad = np.einsum(
            'ij,ikj->ik', dif_between_tetra_nodes, inv_tetra_matrices)

        gradient = ElementData()
        gradient.value = th_grad
        gradient.elm_number = mesh.elm.tetrahedra
        gradient.field_name = 'grad_' + self.field_name

        return gradient

    def append_to_mesh(self, fn, mode='binary'):
        """Appends this NodeData fields to a file

        Parameters:
        --------------------------
            fn: str
                file name
            mode: binary or ascii
                mode in which to write
        """
        with open(fn, 'a') as f:
            f.write('$NodeData' + '\n')
            # string tags
            f.write(str(1) + '\n')
            f.write('"' + self.field_name + '"\n')

            f.write(str(1) + '\n')
            f.write(str(0) + '\n')

            f.write(str(4) + '\n')
            f.write(str(0) + '\n')
            f.write(str(self.nr_comp) + '\n')
            f.write(str(self.nr) + '\n')
            f.write(str(0) + '\n')

            if mode == 'ascii':
                for ii in range(self.nr):
                    f.write(str(self.node_number[
                            ii]) + ' ' + str(self.value[ii]).translate(None, '[](),') + '\n')

            elif mode == 'binary':
                value = self.value.astype('float64')
                try:
                    value.shape[1]
                except IndexError:
                    value = value[:, np.newaxis]

                m = self.node_number[:, np.newaxis].astype('int32')
                for i in range(self.nr_comp):
                    m = np.concatenate((m,
                                        value[:, i].astype('float64').view('int32').reshape(-1, 2)),
                                       axis=1)

                f.write(m.tostring())
            else:
                raise IOError("invalid mode:", mode)

            f.write('$EndNodeData\n')


# =============================================================================
#
# =============================================================================

# 19 Mar 2014 - read meshes with numpy
def read_msh(fn):

    if fn.startswith('~'):
        fn = os.path.expanduser(fn)

    if not os.path.isfile(fn):
        raise IOError(fn + ' not found')

    m = Msh()
    m.fn = fn

    # file open
    f = open(fn, 'rb')

    # check 1st line
    if f.readline() != '$MeshFormat\n':
        raise IOError(fn, "must start with $MeshFormat")

    # parse 2nd line
    version_number, file_type, data_size = f.readline().split()

    if version_number[0] != '2':
        raise IOError("Can only handle v2 meshes")

    if file_type == '1':
        m.binary = True
    elif file_type == '0':
        m.binary = False
    else:
        raise IOError("File_type not recognized:", file_type)

    if data_size != '8':
        raise IOError(
            "data_size should be double (8), i'm reading:", data_size)

    # read next byte, if binary, to check for endianness
    if m.binary:
        endianness = struct.unpack('i', f.readline()[:4])[0]
    else:
        endianness = 1

    if endianness != 1:
        raise RuntimeError("endianness is not 1, is the endian order wrong?")

    # read 3rd line
    if f.readline() != '$EndMeshFormat\n':
        raise IOError(fn + " expected $EndMeshFormat")

    # read 4th line
    if f.readline() != '$Nodes\n':
        raise IOError(fn + " expected $Nodes")

    # read 5th line with number of nodes
    try:
        node_nr = int(f.readline())
    except:
        raise IOError(fn + " something wrong with Line 5 - should be a number")

    # read all nodes
    if m.binary:
        # 0.02s to read binary.msh
        dt = np.dtype([
            ('id', np.int32, 1),
            ('coord', np.float64, 3)])

        temp = np.fromfile(f, dtype=dt, count=node_nr)
        m.nodes.node_number = np.copy(temp['id'])
        m.nodes.node_coord = np.copy(temp['coord'])

        # sometimes there's a line feed here, sometimes there is not...
        LF_byte = f.read(1)  # read \n
        if not ord(LF_byte) == 10:
            # if there was not a LF, go back 1 byte from the current file
            # position
            f.seek(-1, 1)

    else:
        # nodes has 4 entries: [node_ID x y z]
        m.nodes.node_number = np.empty(node_nr, dtype='int32')
        # array Nx3 for (x,y,z) coordinates of the nodes
        m.nodes.node_coord = np.empty(3 * node_nr, dtype='float64')

        # 1.1s for ascii.msh
        for ii in range(node_nr):
            line = f.readline().split()
            m.nodes.node_number[ii] = line[0]
            # it's faster to use a linear array and than reshape
            m.nodes.node_coord[3 * ii] = line[1]
            m.nodes.node_coord[3 * ii + 1] = line[2]
            m.nodes.node_coord[3 * ii + 2] = line[3]
        m.nodes.node_coord = m.nodes.node_coord.reshape((node_nr, 3))

        if len(m.nodes.node_number) != len(m.nodes.node_coord) or \
                m.nodes.node_number[0] != 1 or m.nodes.node_number[-1] != m.nodes.nr:
            raise IOError("Node number is not compact!")

    if f.readline() != '$EndNodes\n':
        raise IOError(fn + " expected $EndNodes after reading " +
                      str(node_nr) + " nodes")

    # read all elements
    if f.readline() != '$Elements\n':
        raise IOError(fn, "expected line with $Elements")

    try:
        elm_nr = int(f.readline())
    except:
        raise IOError(
            fn + " something wrong when reading number of elements (line after $Elements)"
            "- should be a number")

    if m.binary:

        m.elm.elm_number = []
        m.elm.elm_type = []
        m.elm.tag1 = []
        m.elm.tag2 = []
        m.elm.node_number_list = np.zeros((elm_nr, 4), dtype='int32')

        # each element is written in a mini structure that looks like:
        # int header[3] = {elm_type, num_elm_follow, num_tags};
        # then, data for the elementn is written. In the case of a triangle:
        # int data[6] = {num_i, physical, elementary, node_i_1, node_i_2, node_i_3};
        # but for a tetrahedron:
        # int data[7] = {num_i, physical, elementary, node_i_1, node_i_2, node_i_3, node_i_4};
        # ideally, num_elm_follow should be the number of triangles, or tetrahedra, but gmsh
        # always considers this number to be 1, so each element has to be read individually
        # fp=f.tell()
        ii = 0  # start index
        elep = 0  # elements currently processed
        # tt=time.clock()
        nread = 9
        blocktype = 2  # assume triangle first
        binarydata = np.frombuffer(f.read(nread * 4 * elm_nr), dtype='i4')
        while ii < len(binarydata):
            if blocktype == 2:
                nread == 9  # triangle 9 bytes
            elif blocktype == 4:
                # read assuming rest of elements are type 4
                ed = np.frombuffer(f.read((elm_nr - elep) * 4), dtype='i4')
                binarydata = np.hstack((binarydata, ed))
                nread = 10  # tetrahedron 10 bytes
            else:
                raise IOError(
                    "only tetrahedra and triangle supported, but I read type %i" % blocktype)
            # violations of blocktype triangles/tetrahedron
            vlist = np.nonzero(binarydata[ii::nread] != blocktype)[0]
            if len(vlist) == 0:
                iin = len(binarydata[ii::nread])
            else:
                # find first time blocktype is violated triangles/tetrahedron
                iin = np.min(vlist)

            iinn = ii + iin * nread
            m.elm.elm_type.append(binarydata[ii:iinn:nread])
            m.elm.elm_number.append(binarydata[ii + 3:iinn:nread])
            m.elm.tag1.append(binarydata[ii + 4:iinn:nread])
            m.elm.tag2.append(binarydata[ii + 5:iinn:nread])

            if blocktype == 2:
                nd = np.vstack((binarydata[ii + 6:iinn:nread],
                                binarydata[ii + 7:iinn:nread],
                                binarydata[ii + 8:iinn:nread]))
                m.elm.node_number_list[:iin, :3] = nd.T
            if blocktype == 4:
                nd = np.vstack((binarydata[ii + 6:iinn:nread],
                                binarydata[ii + 7:iinn * nread:nread],
                                binarydata[ii + 8:iinn:nread],
                                binarydata[ii + 9:iinn:nread]))
                m.elm.node_number_list[elep:elep + iin, :] = nd.T
            elep += iin
            if iinn < len(binarydata):
                blocktype = binarydata[iinn]
                if blocktype == 2:
                    # reading was too far - rewind
                    f.seek(-(elm_nr - elep) * 4, 1)
                    binarydata = binarydata[:-(elm_nr - elep)]
            ii = iinn
        m.elm.elm_number = np.hstack(m.elm.elm_number)
        m.elm.elm_type = np.hstack(m.elm.elm_type)
        m.elm.tag1 = np.hstack(m.elm.tag1)
        m.elm.tag2 = np.hstack(m.elm.tag2)

        # sometimes there's a line feed here, sometimes there is not...
        LF_byte = f.read(1)  # read \n at end of binary
        if not ord(LF_byte) == 10:
            # if there was not a LF, go back 1 byte from the current file
            # position
            f.seek(-1, 1)

    else:

        m.elm.elm_number = np.empty(elm_nr, dtype='int32')
        m.elm.elm_type = np.empty(elm_nr, dtype='int32')
        m.elm.tag1 = np.empty(elm_nr, dtype='int32')
        m.elm.tag2 = np.empty(elm_nr, dtype='int32')
        m.elm.node_number_list = np.zeros((elm_nr, 4), dtype='int32')

        for ii in range(elm_nr):
            line = f.readline().split()
            m.elm.elm_number[ii] = line[0]
            m.elm.elm_type[ii] = line[1]
            m.elm.tag1[ii] = line[3]
            m.elm.tag2[ii] = line[4]
            if m.elm.elm_type[ii] == 2:
                m.elm.node_number_list[ii, :3] = [int(i) for i in line[5:]]
            elif m.elm.elm_type[ii] == 4:
                m.elm.node_number_list[ii] = [int(i) for i in line[5:]]
            else:
                raise IOError(
                    "ERROR: Meshes must have only triangles and tetrahedra")

    if m.elm.elm_number[0] != 1 or m.elm.elm_number[-1] != m.elm.nr:
        raise IOError("Elements indexes are not compact")

    if f.readline() != '$EndElements\n':
        raise IOError(fn + " expected $EndElements after reading " +
                      str(m.el.nr) + " elements")

    # read the header in the beginning of a data section
    def parse_Data():
        section = f.readline()
        if section == '':
            return 'EOF', '', 0, 0
        # string tags
        number_of_string_tags = int(f.readline())
        assert number_of_string_tags == 1, "Invalid Mesh File: invalid number of string tags"
        name = f.readline().strip().strip('"')
        # real tags
        number_of_real_tags = int(f.readline())
        assert number_of_real_tags == 1, "Invalid Mesh File: invalid number of real tags"
        f.readline()
        # integer tags
        number_of_integer_tags = int(f.readline())  # usually 3 or 4
        integer_tags = [int(f.readline())
                        for i in range(number_of_integer_tags)]
        nr = integer_tags[2]
        nr_comp = integer_tags[1]
        return section.strip(), name, nr, nr_comp

    def read_NodeData(t, name, nr, nr_comp):
        data = NodeData(name=name)
        if m.binary:
            dt = np.dtype([
                ('id', np.int32, 1),
                ('values', np.float64, nr_comp)])

            temp = np.fromfile(f, dtype=dt, count=nr)
            data.node_number = np.copy(temp['id'])
            data.value = np.copy(temp['values'])
            print(data.node_number)
        else:
            data.node_number = np.empty(nr, dtype='int32')
            data.value = np.empty((nr, nr_comp), dtype='float64')
            for ii in range(nr):
                line = f.readline().split()
                data.node_number[ii] = int(line[0])
                data.value[ii, :] = [float(v) for v in line[1:]]

        if f.readline() != '$EndNodeData\n':
            raise IOError(fn + " expected $EndNodeData after reading " +
                          str(nr) + " lines in $NodeData")
        return data

    def read_ElementData(t, name, nr, nr_comp):
        data = ElementData(name=name)
        if m.binary:
            dt = np.dtype([
                ('id', np.int32, 1),
                ('values', np.float64, nr_comp)])

            temp = np.fromfile(f, dtype=dt, count=nr)
            data.elm_number = np.copy(temp['id'])
            data.value = np.copy(temp['values'])

        else:
            data.elm_number = np.empty(nr, dtype='int32')
            data.value = np.empty([nr, nr_comp], dtype='float64')

            for ii in range(nr):
                line = f.readline().split()
                data.elm_number[ii] = int(line[0])
                data.value[ii, :] = [float(jj) for jj in line[1:]]

        if f.readline() != '$EndElementData\n':
            raise IOError(fn + " expected $EndElementData after reading " +
                          str(nr) + " lines in $ElementData")

        return data

    # read sections recursively
    def read_next_section():
        t, name, nr, nr_comp = parse_Data()
        if t == 'EOF':
            return
        elif t == '$NodeData':
            m.nodedata.append(read_NodeData(t, name, nr, nr_comp))
        elif t == '$ElementData':
            m.elmdata.append(read_ElementData(t, name, nr, nr_comp))
        else:
            raise IOError("Can't recognize section name:" + t)

        read_next_section()
        return

    read_next_section()

    return m


# write msh to mesh file
def write_msh(msh, file_name=None, mode='binary'):
    if file_name is not None:
        msh.fn = file_name

    # basic mesh assertions
    if msh.nodes.nr <= 0:
        raise IOError("ERROR: number of nodes is:", msh.nodes.nr)

    if msh.elm.nr <= 0:
        raise IOError("ERROR: number of elements is:", msh.elm.nr)

    if msh.nodes.nr != len(msh.nodes.node_number):
        raise IOError("ERROR: len(nodes.node_number) does not match nodes.nr:",
                      msh.nodes.nr, len(msh.nodes.node_number))

    if msh.nodes.nr != len(msh.nodes.node_coord):
        raise IOError("ERROR: len(nodes.node_coord) does not match nodes.nr:",
                      msh.nodes.nr, len(msh.nodes.node_coord))

    if msh.elm.nr != len(msh.elm.elm_number):
        raise IOError("ERROR: len(elm.elm_number) does not match el.nr:",
                      msh.elm.nr, len(msh.elm.elm_number))

    fn = msh.fn

    if fn[0] == '~':
        fn = os.path.expanduser(fn)

    if mode == 'ascii':
        f = open(fn, 'w')
    elif mode == 'binary':
        f = open(fn, 'wb')
    else:
        raise ValueError("Only 'ascii' and 'binary' are allowed")

    if mode == 'ascii':
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')

    elif mode == 'binary':
        f.write('$MeshFormat\n2.2 1 8\n')
        f.write(struct.pack('i', 1))
        f.write('\n$EndMeshFormat\n')

    # write nodes
    f.write('$Nodes\n')
    f.write(str(msh.nodes.nr) + '\n')

    if mode == 'ascii':
        for ii in range(msh.nodes.nr):
            f.write(str(msh.nodes.node_number[ii]) + ' ' +
                    str(msh.nodes.node_coord[ii][0]) + ' ' +
                    str(msh.nodes.node_coord[ii][1]) + ' ' +
                    str(msh.nodes.node_coord[ii][2]) + '\n')

    elif mode == 'binary':
        node_number = msh.nodes.node_number.astype('int32')
        node_coord = msh.nodes.node_coord.astype('float64')
        f.write(np.concatenate(
            (node_number[:, np.newaxis], node_coord.view('int32')), axis=1).tostring())

    f.write('$EndNodes\n')

    # write elements
    f.write('$Elements\n')
    f.write(str(msh.elm.nr) + '\n')

    if mode == 'ascii':
        for ii in range(msh.elm.nr):
            line = str(msh.elm.elm_number[ii]) + ' ' + \
                str(msh.elm.elm_type[ii]) + ' ' + str(2) + ' ' +\
                str(msh.elm.tag1[ii]) + ' ' + str(msh.elm.tag2[ii]) + ' '

            if msh.elm.elm_type[ii] == 2:
                line += str(msh.elm.node_number_list[ii, :3]
                            ).translate(None, '[](),') + '\n'
            elif msh.elm.elm_type[ii] == 4:
                line += str(msh.elm.node_number_list[ii, :]
                            ).translate(None, '[](),') + '\n'
            else:
                raise IOError(
                    "ERROR: gmsh_numpy cant write meshes with elements of type",
                    msh.elm.elm_type[ii])

            f.write(line)

    elif mode == 'binary':

        triangles = np.where(msh.elm.elm_type == 2)[0]
        triangles_node_list = msh.elm.node_number_list[
            triangles, :3].astype('int32')
        triangles_number = msh.elm.elm_number[triangles].astype('int32')
        triangles_tag1 = msh.elm.tag1[triangles].astype('int32')
        triangles_tag2 = msh.elm.tag2[triangles].astype('int32')
        triangles_ones = np.ones(len(triangles), 'int32')
        triangles_nr_tags = np.ones(len(triangles), 'int32') * 2
        triangles_elm_type = np.ones(len(triangles), 'int32') * 2

        f.write(np.concatenate((triangles_elm_type[:, np.newaxis],
                                triangles_ones[:, np.newaxis],
                                triangles_nr_tags[:, np.newaxis],
                                triangles_number[:, np.newaxis],
                                triangles_tag1[:, np.newaxis],
                                triangles_tag2[:, np.newaxis],
                                triangles_node_list), axis=1).tostring())

        tetra = np.where(msh.elm.elm_type == 4)[0]
        tetra_node_list = msh.elm.node_number_list[tetra].astype('int32')
        tetra_number = msh.elm.elm_number[tetra].astype('int32')
        tetra_tag1 = msh.elm.tag1[tetra].astype('int32')
        tetra_tag2 = msh.elm.tag2[tetra].astype('int32')
        tetra_ones = np.ones(len(tetra), 'int32')
        tetra_nr_tags = np.ones(len(tetra), 'int32') * 2
        tetra_elm_type = np.ones(len(tetra), 'int32') * 4

        f.write(np.concatenate((tetra_elm_type[:, np.newaxis],
                                tetra_ones[:, np.newaxis],
                                tetra_nr_tags[:, np.newaxis],
                                tetra_number[:, np.newaxis],
                                tetra_tag1[:, np.newaxis],
                                tetra_tag2[:, np.newaxis],
                                tetra_node_list), axis=1).tostring())

    f.write('$EndElements\n')
    f.close()

    # write nodeData, if existent
    for nd in msh.nodedata:
        nd.append_to_mesh(fn, mode)

    for eD in msh.elmdata:
        eD.append_to_mesh(fn, mode)


# Adds 1000 to the label of triangles, if less than 100
def create_surface_labels(msh):
    triangles = np.where(msh.elm.elm_type == 2)[0]
    triangles = np.where(msh.elm.tag1[triangles] < 1000)[0]
    msh.elm.tag1[triangles] += 1000
    msh.elm.tag2[triangles] += 1000
    return msh


def read_res_file(fn):
    """ Reads a .res file

    Parameters
    -----------------------------
    fn: str
        name of fime

    Returns
    -----------------------------
    ndarray
        values
    """
    with open(fn, 'r') as f:
        f.readline()  # skip first line
        check, type_of_file = f.readline().strip('\n').split(' ')

        if check != '1.1':
            raise IOError('Unexpected value in res!')

        if type_of_file == '0':
            v = np.loadtxt(f, comments='$', skiprows=3, usecols=[
                           0], delimiter=' ', dtype='float64')

        elif type_of_file == '1':
            s = ''.join(f.readlines()[3:-1])
            cols = np.fromstring(s[0:-1], dtype=np.dtype('float64'))
            v = cols[::2]

        else:
            raise IOError(
                'Do not recognize file type: %s for res file' % type_of_file)

    return v

def write_geo_spheres(positions, fn, values=None, name="", size=7):
    """ Writes a .geo file with spheres in specified positions

    Parameters:
    ------------
    positions: nx3 ndarray:
        position of spheres
    fn: str
        name of file to be written
    values(optional): nx1 ndarray
        values to be assigned to the spheres. Default: 0
    name(optional): str
        Name of the view
    size: float
        Size of the sphere

    Returns:
    ------------
    writes the "fn" file
    """
    if values is None:
        values = np.zeros((len(positions), ))

    if len(values) != len(positions):
        raise ValueError('The length of the vector of positions is different from the'
                         'length of the vector of values')

    with open(fn, 'w') as f:
        f.write('View"' + name + '"{\n')
        for p, v in zip(positions, values):
            f.write("SP(" + ", ".join([str(i) for i in p]) + "){" + str(v) + "};\n")
        f.write("};\n")
        f.write("myView = PostProcessing.NbViews-1;\n")
        f.write("View[myView].PointType=1; // spheres\n")
        f.write("View[myView].PointSize=" + str(size) + ";")
