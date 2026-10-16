from setting import*
from numba import uint8

@njit
def get_ao(local_pos,world_pos,world_voxels,plane):
    x,y,z=local_pos
    wx,wy,wz=world_pos

    if plane=='Y':
        a=is_void((x,y,z-1),(wx,wy,wz-1),world_voxels)
        b=is_void((x-1,y,z-1),(wx-1,wy,wz-1),world_voxels)
        c=is_void((x-1,y,z),(wx-1,wy,wz),world_voxels)
        d=is_void((x-1,y,z+1),(wx-1,wy,wz+1),world_voxels)
        e=is_void((x,y,z+1),(wx,wy,wz+1),world_voxels)
        f=is_void((x+1,y,z+1),(wx+1,wy,wz+1),world_voxels)
        g=is_void((x+1,y,z),(wx+1,wy,wz),   world_voxels)
        h=is_void((x+1,y,z-1),(wx+1,wy,wz-1),world_voxels)

    elif plane=='X':
        a=is_void((x,y,z-1),(wx,wy,wz-1),world_voxels)
        b=is_void((x,y-1,z-1),(wx,wy-1,wz-1),world_voxels)
        c=is_void((x,y-1,z),(wx,wy-1,wz),world_voxels)
        d=is_void((x,y-1,z+1),(wx,wy-1,wz+1),world_voxels)
        e=is_void((x,y,z+1),(wx,wy,wz+1),world_voxels)
        f=is_void((x,y+1,z+1),(wx,wy+1,wz+1),world_voxels)
        g=is_void((x,y+1,z),(wx,wy+1,wz),   world_voxels)
        h=is_void((x,y+1,z-1),(wx,wy+1,wz-1),world_voxels)

    else: #Z plane
        a=is_void((x-1,y,z),(wx-1,wy,wz),world_voxels)
        b=is_void((x-1,y-1,z),(wx-1,wy-1,wz),world_voxels)
        c=is_void((x,y-1,z),(wx,wy-1,wz),world_voxels)
        d=is_void((x+1,y-1,z),(wx+1,wy-1,wz),world_voxels)
        e=is_void((x+1,y,z),(wx+1,wy,wz),world_voxels)
        f=is_void((x+1,y+1,z),(wx+1,wy+1,wz),world_voxels)
        g=is_void((x,y+1,z),(wx,wy+1,wz),   world_voxels)
        h=is_void((x-1,y+1,z),(wx-1,wy+1,wz),world_voxels)

    ao=(a+b+c),(g+h+a),(e+f+g),(c+d+e)
    return ao

@njit
def to_uint8(x,y,z,voxel_id,face_id,ao_id,flip_id):
    return (uint8(x),uint8(y),uint8(z),uint8(voxel_id),uint8(face_id),uint8(ao_id),uint8(flip_id))

@njit
def pack_data(x,y,z,voxel_id,face_id,ao_id,flip_id):
    a,b,c,d,e,f,g=x,y,z,voxel_id,face_id,ao_id,flip_id

    b_bit,c_bit,d_bit,e_bit,f_bit,g_bit=6,6,8,3,2,1
    fg_bit=f_bit+g_bit
    efg_bit=e_bit+fg_bit
    defg_bit=d_bit+efg_bit
    cdefg_bit=c_bit+defg_bit
    bcdefg_bit=b_bit+cdefg_bit

    packed_data=(
        a<<bcdefg_bit|
        b<<cdefg_bit|
        c<<defg_bit|
        d<<efg_bit|
        e<<fg_bit|
        f<<g_bit| g
    )
    return packed_data

@njit
def get_chunk_index(world_voxel_pos):
    wx,wy,wz=world_voxel_pos
    cx,cy,cz=wx//CHUNK_SIZE,wy//CHUNK_SIZE,wz//CHUNK_SIZE
    if not(0<=cx<WORLD_W and 0<=cy<WORLD_H and 0<=cz<WORLD_D):
     return -1
    
    index=cx+WORLD_W*cz+WORLD_AREA*cy
    return index

@njit
def is_void(local_voxel_pos,world_voxel_pos,world_voxels):
    chunk_index=get_chunk_index(world_voxel_pos)
    if chunk_index==-1:
        return False
    chunk_voxels=world_voxels[chunk_index]

    x,y,z=local_voxel_pos
    voxel_index=x%CHUNK_SIZE+z%CHUNK_SIZE*CHUNK_SIZE+y%CHUNK_SIZE*CHUNK_AREA

    if chunk_voxels[voxel_index]:
        return False
    return True

@njit
def add_data(vertex_data,index,*vertices):
    for vertex in vertices:
            vertex_data[index]=vertex
            index+=1
    return index    
    

@njit
def build_chunk_mesh(chunk_voxels,format_size,chunk_pos,world_voxels):
    vertex_data = np.empty(CHUNK_VOL * 36 * format_size, dtype=np.uint32)
    index=0

    for x in range(CHUNK_SIZE):
        for y in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                voxel_id=chunk_voxels[x+CHUNK_SIZE*z+CHUNK_AREA*y]
                if not voxel_id:
                    continue

                #voxel world position
                cx,cy,cz=chunk_pos
                wx=x+cx*CHUNK_SIZE
                wy=y+cy*CHUNK_SIZE
                wz=z+cz*CHUNK_SIZE

                #top face
                if face_visible((x,y+1,z),(wx,wy+1,wz),world_voxels,voxel_id):
                    #get ao values
                    ao=get_ao((x,y+1,z),(wx,wy+1,wz),world_voxels,plane='Y')
                    flip_id=ao[1]+ao[3]>ao[0]+ao[2]

                    #format:x,y,z voxel_id,face_id,ao_id
                    v0=pack_data(x,y+1,z,voxel_id,0,ao[0],flip_id)
                    v1=pack_data(x+1,y+1,z,voxel_id,0,ao[1],flip_id)
                    v2=pack_data(x+1,y+1,z+1,voxel_id,0,ao[2],flip_id)
                    v3=pack_data(x ,y+1,z+1,voxel_id,0,ao[3],flip_id)

                    if flip_id:
                        index=add_data(vertex_data,index,v1,v0,v3,v1,v3,v2)
                    else:
                        index=add_data(vertex_data,index,v0,v3,v2,v0,v2,v1)
                   #bottom face
                if face_visible((x,y-1,z),(wx,wy-1,wz),world_voxels,voxel_id):   

                    ao=get_ao((x,y-1,z),(wx,wy-1,wz),world_voxels,plane='Y')
                    flip_id=ao[1]+ao[3]>ao[0]+ao[2]


                    v0=pack_data(x,y,z,voxel_id,1,ao[0],flip_id)
                    v1=pack_data(x+1,y,z,voxel_id,1,ao[1],flip_id)
                    v2=pack_data(x+1,y,z+1,voxel_id,1,ao[2],flip_id)
                    v3=pack_data(x,y,z+1,voxel_id,1,ao[3],flip_id)

                    if flip_id:
                        index=add_data(vertex_data,index,v1,v0,v3,v1,v3,v2)
                    else:
                        index=add_data(vertex_data,index,v0,v3,v2,v0,v2,v1)

                #right face
                if face_visible((x+1,y,z),(wx+1,wy,wz),world_voxels,voxel_id):

                    ao=get_ao((x+1,y,z),(wx+1,wy,wz),world_voxels,plane='X')
                    flip_id=ao[1]+ao[3]>ao[0]+ao[2]

                    v0=pack_data(x+1,y,z,voxel_id,2,ao[0],flip_id)
                    v1=pack_data(x+1,y+1,z,voxel_id,2,ao[1],flip_id)
                    v2=pack_data(x+1,y+1,z+1,voxel_id,2,ao[2],flip_id)
                    v3=pack_data(x+1,y,z+1,voxel_id,2,ao[3],flip_id)

                    if flip_id:
                        index=add_data(vertex_data,index,v1,v0,v3,v1,v3,v2)
                    else:
                        index=add_data(vertex_data,index,v0,v3,v2,v0,v2,v1)

                #left face
                if face_visible((x-1,y,z),(wx-1,wy,wz),world_voxels,voxel_id):

                    ao=get_ao((x-1,y,z),(wx-1,wy,wz),world_voxels,plane='X')
                    flip_id=ao[1]+ao[3]>ao[0]+ao[2]

                    v0=pack_data(x,y,z,voxel_id,3,ao[0],flip_id)
                    v1=pack_data(x,y+1,z,voxel_id,3,ao[1],flip_id)
                    v2=pack_data(x,y+1,z+1,voxel_id,3,ao[2],flip_id)
                    v3=pack_data(x,y,z+1,voxel_id,3,ao[3],flip_id)

                    if flip_id:
                        index = add_data(
                                         vertex_data,
                                        index,
                                         v1, v0, v3, v1, v3, v2)
                    else:
                        index = add_data(
                                         vertex_data,
                                        index,
                                         v0, v3, v2, v0, v2, v1)

                #back face
                if face_visible((x,y,z-1),(wx,wy,wz-1),world_voxels,voxel_id):

                    ao=get_ao((x,y,z-1),(wx,wy,wz-1),world_voxels,plane='Z')
                    flip_id=ao[1]+ao[3]>ao[0]+ao[2]

                    v0=pack_data(x,y,z,voxel_id,4,ao[0],flip_id)
                    v1=pack_data(x,y+1,z,voxel_id,4,ao[1],flip_id)
                    v2=pack_data(x+1,y+1,z,voxel_id,4,ao[2],flip_id)
                    v3=pack_data(x+1,y,z,voxel_id,4,ao[3],flip_id)

                    if flip_id:
                        index=add_data(vertex_data,index,v1,v0,v3,v1,v3,v2)
                    else:
                        index=add_data(vertex_data,index,v0,v3,v2,v0,v2,v1)

                #front face
                if face_visible((x,y,z+1),(wx,wy,wz+1),world_voxels,voxel_id):

                    ao=get_ao((x,y,z+1),(wx,wy,wz+1),world_voxels,plane='Z')
                    flip_id=ao[1]+ao[3]>ao[0]+ao[2]

                    v0=pack_data(x,y,z+1,voxel_id,5,ao[0],flip_id)
                    v1=pack_data(x,y+1,z+1,voxel_id,5,ao[1],flip_id)
                    v2=pack_data(x+1,y+1,z+1,voxel_id,5,ao[2],flip_id)
                    v3=pack_data(x+1,y,z+1,voxel_id,5,ao[3],flip_id)

                    if flip_id:
                        index=add_data(vertex_data,index,v1,v0,v3,v1,v3,v2)
                    else:
                        index=add_data(vertex_data,index,v0,v3,v2,v0,v2,v1)

    return vertex_data[:index]
@njit
def face_visible(local_pos, world_pos, world_voxels, voxel_id):
    chunk_index = get_chunk_index(world_pos)
    if chunk_index == -1:
        return False
    chunk_voxels = world_voxels[chunk_index]

    x, y, z = local_pos
    n = chunk_voxels[x % CHUNK_SIZE + z % CHUNK_SIZE * CHUNK_SIZE + y % CHUNK_SIZE * CHUNK_AREA]

    if n == 0:
        return True                      # hawa: face dikhao
    if n == 8 and voxel_id != 8:
        return True                      # terrain ke saamne paani: face dikhao (bed)
    return False                         # paani-paani ya solid: nahi













