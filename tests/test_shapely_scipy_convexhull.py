
#------------------------------------------------------------
# x,y coords == x,y components of vector (obviously)
#------------------------------------------------------------

ref_frames = {'neu': {'origin': np.array([[0,0]]), 'ref_vec': np.array([[6,0]]), 'angle_drn': False},
              'pov': {'origin': np.array([[6,0]]), 'ref_vec': np.array([[6,6]]), 'angle_drn': None}} 

ref_frame = ref_frames['neu']
dists   = ComputeBehavior2.calc_distance(coords, ref_frame['origin'])
angles  = ComputeBehavior2.calc_angle(coords, ref_frame, n_dim=2)
xy_comp = polar_to_vec_comps(angles, dists)
np.sum(np.round(xy_comp, 4) == coords) == 126


#------------------------------------------------------------
# convex hull from scipy & shapely
#------------------------------------------------------------

from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import MultiPoint, mapping

def test_convexhulls(coords):

    # scipy implementation: 
    convexhull_scipy = ConvexHull(coords) 
    vertices_scipy   = coords[convexhull_scipy.vertices]
    area_scipy       = convexhull_scipy.volume

    # shapely
    convexhull_shply = Polygon(coords).convex_hull
    vertices_shply   = np.array(mapping(convexhull_shply)['coordinates'][0]) # gives the points that make up the convex hull
    area_shply       = convexhull_shply.area


    if np.round(area_scipy,5) != np.round(area_shply,5): 
        
        print(f'The areas dont match: scipy {area_scipy} != shapely {area_shply}')

        # then check shapes:
        if (vertices_scipy.shape != vertices_shply.shape):
            print(f'The shapes dont match: scipy={vertices_scipy.shape}, shapely={vertices_shply.shape}')

            # wheres the difference?
            vertices_to_tuple = lambda vertices: [tuple(v.tolist()) for v in vertices]
            set_to_array = lambda set_: np.array(list(set_))

            # this will remove redundant points
            scipy_set = set(vertices_to_tuple(vertices_scipy))
            shply_set = set(vertices_to_tuple(vertices_shply))

            intersection = list(scipy_set & shply_set)
            union = list(scipy_set | shply_set)
            diff  = scipy_set ^ shply_set 
            if len(diff) == 0: print('Shapely captured same point twice')

    else: 
        print('The areas are the same')