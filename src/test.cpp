/*
 * defintions
 *
 * aiBone is a bone. its purpose is only for skinning. contains:
 *      name of the aiNode it corresponds to
 *      which vertices it influences and their weights
 *      offset matrix that brings mesh space -> bone space
 *
 *  note that different meshes can share the same bone name, though with different offset matrices, because the offset is relative to that mesh's space.
 *
 *
 * aiNodeAnim (aka channel) is an animated node. it has keyframes. only exists for nodes that have animations. contains:
 *      name of the aiNode it corresponds to
 *      three keyframe arrays
 *
 * aiNode is just a boring node. defines hierarchy. contains:
 *      a name
 *      a local transform matrix relative to its parent in bind pose
 *      indices to meshes
 *      parents and children
 *
 *  if the aiNode has no corresponding aiNodeAnim, just use the default local transform matrix in bind pose.
 *
 *
 *
 * final bone matrix: global_inverse * global_transform * offset_matrix(or inverse bind matrix)
 * global_inverse: root node's transform inversed
 * global_transform: walk down aiNode tree. if node is animated, use T * R * S animated local transform. otherwise use its node local transform in bind pose.
 * inverse bind matrix: a.k.a offset matrix. mesh space -> bone space. from aiBone::mOffsetMatrix
 */
#include "assimp/mesh.h"
#include "assimp/quaternion.h"
#include "assimp/vector3.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <raylib.h>
#include <raymath.h>

namespace test {
    struct Vertex {
        Eigen::Vector3f position = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal = Eigen::Vector3f::Zero();
        Eigen::Vector2f texture_coordinates = Eigen::Vector2f::Zero();
            
        // currently at most 4 bones affect a vertex at once
        int bone_ids[4]; // the ids of bones which will affect this vertex
        float bone_weights[4];
    };

    struct KeyframePosition {
        Eigen::Vector3f position;
        float timestamp;
    };

    struct KeyframeRotation {
        Eigen::Quaternion<float> rotation;
        float timestamp;
    };

    struct KeyframeScale {
        Eigen::Vector3f scale;
        float timestamp;
    };

    // a bone: an aiNode that influences vertices. created from aiBone
    struct bone_info {
        int id;
        Eigen::Matrix4<float> offset_matrix; // a.k.a inverse bind matrix. transforms from model space to bone space, in bind pose. static matrix
    };

    // an aiNode that has keyframes in the animation. not necessarily a bone (see bone_info)
    class AnimatedNode {
        public:
            std::vector<KeyframePosition> positions;
            std::vector<KeyframeRotation> rotations;
            std::vector<KeyframeScale> scales;
            int num_positions;
            int num_rotations;
            int num_scales;

            std::string name;
            int bone_id; // id in Model.bone_map, -1 if this node is not a bone
            Eigen::Matrix4<float> local_matrix; // translate * rotate * scale. updates every frame

            // reads keyframes from channel
            // a channel is a node's oritentation data at various times
            AnimatedNode(std::string name, int bone_id, aiNodeAnim* channel) {
                this->name = name;
                this->bone_id = bone_id;
                
                num_positions = channel->mNumPositionKeys;
                for (int i = 0; i < num_positions; i++) {
                    aiVector3d aiPosition = channel->mPositionKeys[i].mValue;
                    float timestamp = channel->mPositionKeys[i].mTime;
                    
                    KeyframePosition my_position;
                    my_position.position = Eigen::Vector3f(aiPosition.x, aiPosition.y, aiPosition.z);
                    my_position.timestamp = timestamp;

                    positions.push_back(my_position);
                }

                num_rotations = channel->mNumRotationKeys;
                for (int i = 0; i < num_rotations; i++) {
                    aiQuaternion aiQuat = channel->mRotationKeys[i].mValue; 
                    float timestamp = channel->mRotationKeys[i].mTime;

                    KeyframeRotation my_rotation;
                    my_rotation.rotation = Eigen::Quaternion(aiQuat.w, aiQuat.x, aiQuat.y, aiQuat.z);
                    my_rotation.timestamp = timestamp;

                    rotations.push_back(my_rotation);
                }

                num_scales = channel->mNumScalingKeys;
                for (int i = 0; i < num_scales; i++) {
                    aiVector3d aiScale = channel->mScalingKeys[i].mValue;
                    float timestamp = channel->mScalingKeys[i].mTime;

                    KeyframeScale my_scale;
                    my_scale.scale = Eigen::Vector3f(aiScale.x, aiScale.y, aiScale.z);
                    my_scale.timestamp = timestamp;

                    scales.push_back(my_scale);
                }
            }

            // gets the scale factor required for lerp/slerp between two keyframes
            float get_scale_factor(float last_timestamp, float next_timestamp, float current_timestamp) {
                float difference_from_last_timestamp = current_timestamp - last_timestamp; // delta time from the current timestamp to the last keyframe's timestamp
                float time_between_keyframe = next_timestamp - last_timestamp; // delta time from the last keyframe's timestamp to the next keyframe's timestamp
                return difference_from_last_timestamp / time_between_keyframe;
            }

            Eigen::Affine3f interpolate_position(float current_timestamp) {
                if (num_positions == 1) {
                    return Eigen::Affine3f(Eigen::Translation3f(this->positions[0].position));
                }

                int p1 = num_positions - 2;
                for (int i = 0; i < num_positions - 1; i++) {
                    if (current_timestamp < this->positions[i + 1].timestamp) {
                        p1 = i;
                        break;
                    }
                }
                int p2 = p1 + 1;

                float scale_factor = get_scale_factor(this->positions[p1].timestamp, this->positions[p2].timestamp, current_timestamp);
                const Eigen::Vector3f& a = this->positions[p1].position;
                const Eigen::Vector3f& b = this->positions[p2].position;
                Eigen::Vector3f final_position = a + (b - a) * scale_factor; // lerp
                return Eigen::Affine3f(Eigen::Translation3f(final_position));
            }

            Eigen::Affine3f interpolate_rotation(float current_timestamp) {
                if (num_rotations == 1) {
                    return Eigen::Affine3f(this->rotations[0].rotation.normalized());
                }

                int p1 = num_rotations - 2;
                for (int i = 0; i < num_rotations - 1; i++) {
                    if (current_timestamp < this->rotations[i + 1].timestamp) {
                        p1 = i;
                        break;
                    }
                }
                int p2 = p1 + 1;

                float scale_factor = get_scale_factor(this->rotations[p1].timestamp, this->rotations[p2].timestamp, current_timestamp);
                const Eigen::Quaternionf& a = this->rotations[p1].rotation;
                const Eigen::Quaternionf& b = this->rotations[p2].rotation;
                Eigen::Quaternionf final_rotation = a.slerp(scale_factor, b).normalized(); // slerp
                return Eigen::Affine3f(final_rotation);
            }

            Eigen::Affine3f interpolate_scale(float current_timestamp) {
                if (num_scales == 1) {
                    return Eigen::Affine3f(Eigen::AlignedScaling3f(this->scales[0].scale));
                }

                int p1 = num_scales - 2;
                for (int i = 0; i < num_scales - 1; i++) {
                    if (current_timestamp < this->scales[i + 1].timestamp) {
                        p1 = i;
                        break;
                    }
                }
                int p2 = p1 + 1;

                float scale_factor = get_scale_factor(this->scales[p1].timestamp, this->scales[p2].timestamp, current_timestamp);
                const Eigen::Vector3f& a = this->scales[p1].scale;
                const Eigen::Vector3f& b = this->scales[p2].scale;
                Eigen::Vector3f final_scale = a + (b - a) * scale_factor; // lerp
                return Eigen::Affine3f(Eigen::AlignedScaling3f(final_scale));
            }

            // sets node's local matrix, updates every frame
            void update(float current_timestamp) {
                // converting to matrix here might be redundant. consider just keeping affine3f all the way
               local_matrix = (interpolate_position(current_timestamp) * interpolate_rotation(current_timestamp) * interpolate_scale(current_timestamp)).matrix();
            }
        
    };

    // a model should have a loaded texture array for optimization
    struct Texture {
        unsigned int id;
        std::string type;
        std::string path;
    };

    class Mesh {
        public:
            std::vector<Vertex> vertices;
            std::vector<unsigned int> indices; // indices to save memory due to lots of vertices reused
            std::vector<Texture> textures; // TODO
    };

    class Model {
        public:
            std::vector<Mesh> meshes;
            std::unordered_map<std::string, bone_info> bone_map; // maps a bone's name to the bone's information
            int bone_count = 0; // also the id given to the next new bone
            void processNode(aiNode* assimp_node, const aiScene *assimp_scene);
            void processMesh(aiNode* assimp_node, const aiScene *assimp_scene);
    };

    struct assimp_node {
        Eigen::Matrix4f transformation; // matrix that represents position, rotation, and orientation relative to its parent, in BIND POSE
        std::string name;
        std::vector<assimp_node> children;
    };

    assimp_node read_hierarchy_data(aiNode* current_node) {
        assimp_node my_node;
        my_node.transformation = Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(&current_node->mTransformation.a1);
        my_node.name = current_node->mName.data;

        for (unsigned int i = 0; i < current_node->mNumChildren; i++) {
            my_node.children.push_back(read_hierarchy_data(current_node->mChildren[i]));
        }

        return my_node;
    }
}

/*
 * populates Model.meshes field
 */
void test::Model::processMesh(aiNode* assimp_node, const aiScene *assimp_scene) {
    for (unsigned int i = 0; i < assimp_node->mNumMeshes; i++) {
        // scene owns the actual mesh object, node owns the indices to the scene's mesh objects
        aiMesh* mesh = assimp_scene->mMeshes[assimp_node->mMeshes[i]];

        test::Mesh my_mesh;
        std::vector<test::Vertex> current_mesh_vertices;
        std::vector<unsigned int> current_mesh_indices;
        std::vector<test::Texture> current_mesh_textures; // TODO

        // create vertices attribute from assimp types
        for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
            test::Vertex my_vertex;

            my_vertex.position = Eigen::Vector3f(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
            my_vertex.normal = Eigen::Vector3f(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);
            if (mesh->mTextureCoords[0]) { // if the mesh actually has texture coordinates
                my_vertex.texture_coordinates = Eigen::Vector2f(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y);
            } else {
                my_vertex.texture_coordinates = Eigen::Vector2f(0, 0);
            }

            // set the vertex's bone data to default
            for (int k = 0; k < 4; k++) {
                my_vertex.bone_ids[k] = -1;
                my_vertex.bone_weights[k] = 0.0f;
            }

            current_mesh_vertices.push_back(my_vertex);
        }

        
        // create indices attribute from assimp types
        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            aiFace current_face = mesh->mFaces[j]; // a triangle with 3 indices
            for (unsigned int k = 0; k < current_face.mNumIndices; k++) {
                current_mesh_indices.push_back(current_face.mIndices[k]);
            }
        }


        // TODO: support materials


        // iterate through bones that will influence this mesh
        // note this isn't the bone itself, and thus multiple meshes can be influenced by the same bone
        for (unsigned int j = 0; j < mesh->mNumBones; j++) {
            aiBone* bone = mesh->mBones[j];
            std::string bone_name = bone->mName.C_Str();
            int current_bone_id;

            if (this->bone_map.find(bone_name) == bone_map.end()) { // if bone has not been created/registered yet
                test::bone_info new_bone;

                new_bone.id = this->bone_count++;
                new_bone.offset_matrix = Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(&bone->mOffsetMatrix.a1); // black magic that turns aiMatrix4x4 to Eigen::Matrix4f
                    
                this->bone_map[bone_name] = new_bone;

                current_bone_id = new_bone.id;
            } else {
                current_bone_id = this->bone_map[bone_name].id;
            }

            // iterate through each vertex that is influenced by this bone
            for (unsigned int k = 0; k < bone->mNumWeights; k++) {
                int vertex_id = bone->mWeights[k].mVertexId; // vertex index within the mesh scope, NOT model scope
                float weight = bone->mWeights[k].mWeight; // the influence/weight this bone has on the vertex

                assert(vertex_id < (int) current_mesh_vertices.size());

                for (int l = 0; l < 4; l++) {
                    if (current_mesh_vertices[vertex_id].bone_ids[l] == -1) { // unregistered bone position
                        current_mesh_vertices[vertex_id].bone_ids[l] = current_bone_id;
                        current_mesh_vertices[vertex_id].bone_weights[l] = weight;
                        break;
                    }
                }
            }
        }


        my_mesh.vertices = current_mesh_vertices;
        my_mesh.indices = current_mesh_indices;
        my_mesh.textures = current_mesh_textures;

        this->meshes.push_back(my_mesh);
    }
}

/*
 * recursive function that traverses through the aiScene's nodes
 */
void test::Model::processNode(aiNode* assimp_node, const aiScene *assimp_scene) {
    processMesh(assimp_node, assimp_scene);

    // recursively process node's children
    for (unsigned int i = 0; i < assimp_node->mNumChildren; i++) {
        processNode(assimp_node->mChildren[i], assimp_scene);
    }
}

static Mesh transform_to_raylib_mesh(const test::Mesh &my_mesh) {
    Mesh raylib_mesh = {};
    raylib_mesh.vertexCount = my_mesh.vertices.size();
    raylib_mesh.triangleCount = my_mesh.indices.size() / 3;

    // UnloadMesh frees these with MemFree, so allocate them with MemAlloc
    raylib_mesh.vertices = (float *) MemAlloc(raylib_mesh.vertexCount * 3 * sizeof(float));
    raylib_mesh.colors = (unsigned char *) MemAlloc(raylib_mesh.vertexCount * 4);
    raylib_mesh.indices = (unsigned short *) MemAlloc(my_mesh.indices.size() * sizeof(unsigned short));

    for (int j = 0; j < raylib_mesh.vertexCount; j++) {
        const test::Vertex &my_vertex = my_mesh.vertices[j];

        raylib_mesh.vertices[j * 3 + 0] = my_vertex.position.x();
        raylib_mesh.vertices[j * 3 + 1] = my_vertex.position.y();
        raylib_mesh.vertices[j * 3 + 2] = my_vertex.position.z();

        unsigned char shade = (unsigned char)(128.0f + 127.0f * my_vertex.normal.y());
        raylib_mesh.colors[j * 4 + 0] = shade;
        raylib_mesh.colors[j * 4 + 1] = shade;
        raylib_mesh.colors[j * 4 + 2] = shade;
        raylib_mesh.colors[j * 4 + 3] = 255;
    }

    // raylib's index buffer is 16 bit, so a mesh past 65535 vertices would wrap
    for (size_t j = 0; j < my_mesh.indices.size(); j++) {
        raylib_mesh.indices[j] = (unsigned short)my_mesh.indices[j];
    }

    UploadMesh(&raylib_mesh, false);
    return raylib_mesh;
}

/*
 * recursive function that walks down the node tree, calculating each bone's final bone matrix
 */
static void calculate_bone_matrices(const test::assimp_node &node, const Eigen::Matrix4f &parent_transform, float current_timestamp,
                                    std::unordered_map<std::string, test::AnimatedNode> &animated_nodes, test::Model &my_model,
                                    const Eigen::Matrix4f &global_inverse, std::vector<Eigen::Matrix4f> &final_bone_matrices) {
    Eigen::Matrix4f node_transform = node.transformation; // node local transform in bind pose
    auto animated_node = animated_nodes.find(node.name);
    if (animated_node != animated_nodes.end()) { // node is animated, use animated local transform instead
        animated_node->second.update(current_timestamp);
        node_transform = animated_node->second.local_matrix;
    }

    Eigen::Matrix4f global_transform = parent_transform * node_transform;

    auto bone = my_model.bone_map.find(node.name);
    if (bone != my_model.bone_map.end()) { // node is a bone that influences vertices
        final_bone_matrices[bone->second.id] = global_inverse * global_transform * bone->second.offset_matrix;
    }

    // recursively process node's children
    for (const test::assimp_node &child : node.children) {
        calculate_bone_matrices(child, global_transform, current_timestamp, animated_nodes, my_model, global_inverse, final_bone_matrices);
    }
}

int main() {
    Assimp::Importer importer;
    // ../Walking.fbx is relative to the path you launch the program from
    const aiScene *scene = importer.ReadFile("Walking.fbx", aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_LimitBoneWeights);
    if (scene == nullptr) {
        std::cout << importer.GetErrorString() << std::endl;
        return 1;
    }

    test::Model my_model;
    my_model.processNode(scene->mRootNode, scene);

    auto animation = scene->mAnimations[0];
    float duration = animation->mDuration;
    int ticks_per_second = animation->mTicksPerSecond;

    // create an AnimatedNode for every node with a channel
    // channels can also animate nodes that aren't bones (e.g. assimp's $AssimpFbx$ helper nodes), those get bone id -1
    std::unordered_map<std::string, test::AnimatedNode> animated_nodes; // maps a node's name to its keyframes
    for (unsigned int i = 0; i < animation->mNumChannels; i++) {
        aiNodeAnim* channel = animation->mChannels[i];
        std::string node_name = channel->mNodeName.C_Str();
        int current_bone_id = -1;

        if (my_model.bone_map.find(node_name) != my_model.bone_map.end()) {
            current_bone_id = my_model.bone_map[node_name].id;
        }

        animated_nodes.emplace(node_name, test::AnimatedNode(node_name, current_bone_id, channel));
    }

    test::assimp_node root_node = test::read_hierarchy_data(scene->mRootNode);
    Eigen::Matrix4f global_inverse = root_node.transformation.inverse();

    std::vector<Eigen::Matrix4f> final_bone_matrices(my_model.bone_count,Eigen::Matrix4f::Identity());


    // === RAYLIB STUFF ===

    InitWindow(1280, 720, "test");
    SetTargetFPS(60);

    std::vector<Mesh> raylib_meshes;
    for (const test::Mesh &my_mesh : my_model.meshes) {
        raylib_meshes.push_back(transform_to_raylib_mesh(my_mesh));
    }

    Material material = LoadMaterialDefault();

    Camera3D camera = {};
    camera.position = { 0.0f, 100.0f, 300.0f };
    camera.target = { 0.0f, 90.0f, 0.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    while (!WindowShouldClose()) {
        UpdateCamera(&camera, CAMERA_ORBITAL);

        // === ANIMATION ===

        float current_timestamp = fmod(GetTime() * ticks_per_second, duration); // in ticks, loops the animation

        // walk down the node tree to calculate each bone's final bone matrix
        calculate_bone_matrices(root_node, Eigen::Matrix4f::Identity(), current_timestamp, animated_nodes, my_model, global_inverse, final_bone_matrices);

        // linear blend skinning on the cpu, then reupload the vertex positions
        for (size_t i = 0; i < my_model.meshes.size(); i++) {
            const test::Mesh& my_mesh = my_model.meshes[i];
            Mesh& raylib_mesh = raylib_meshes[i];

            for (size_t j = 0; j < my_mesh.vertices.size(); j++) {
                const test::Vertex& my_vertex = my_mesh.vertices[j];
                Eigen::Vector4f bind_position = my_vertex.position.homogeneous(); // (x, y, z, 1) so translation applies
                Eigen::Vector4f final_position = Eigen::Vector4f::Zero();

                for (int k = 0; k < 4; k++) {
                    if (my_vertex.bone_ids[k] == -1) continue;
                    final_position += my_vertex.bone_weights[k] * (final_bone_matrices[my_vertex.bone_ids[k]] * bind_position);
                }

                if (my_vertex.bone_ids[0] == -1) { // vertex not influenced by any bone, keep bind pose
                    final_position = bind_position;
                }

                raylib_mesh.vertices[j * 3 + 0] = final_position.x();
                raylib_mesh.vertices[j * 3 + 1] = final_position.y();
                raylib_mesh.vertices[j * 3 + 2] = final_position.z();
            }

            UpdateMeshBuffer(raylib_mesh, 0, raylib_mesh.vertices, raylib_mesh.vertexCount * 3 * sizeof(float), 0);
        }

        BeginDrawing();
        ClearBackground(GRAY);

        BeginMode3D(camera);
        for (Mesh &raylib_mesh : raylib_meshes) DrawMesh(raylib_mesh, material, MatrixIdentity());
        EndMode3D();

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
