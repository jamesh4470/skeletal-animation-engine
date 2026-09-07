#include "assimp/vector3.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <string>
#include <vector>

#include <raylib.h>
#include <raymath.h>

namespace test {
    struct Vertex {
        Eigen::Vector3f position = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal = Eigen::Vector3f::Zero();
        Eigen::Vector2f texture_coordinates = Eigen::Vector2f::Zero();
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
            void processNode(aiNode* assimp_node, const aiScene *assimp_scene);
    };
}

void test::Model::processNode(aiNode* assimp_node, const aiScene *assimp_scene) {
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

            current_mesh_vertices.push_back(my_vertex);
        }

        
        // create indices attribute from assimp types
        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            aiFace current_face = mesh->mFaces[j]; // a triangle
            for (unsigned int k = 0; k < current_face.mNumIndices; k++) {
                current_mesh_indices.push_back(current_face.mIndices[k]);
            }
        }


        // TODO: support materials

        my_mesh.vertices = current_mesh_vertices;
        my_mesh.indices = current_mesh_indices;
        my_mesh.textures = current_mesh_textures;

        meshes.push_back(my_mesh);
    }

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

int main() {
    Assimp::Importer importer;
    // ../Walking.fbx is relative to the path you launch the program from
    const aiScene *scene = importer.ReadFile("Walking.fbx", aiProcess_Triangulate | aiProcess_FlipUVs);

    if (scene == nullptr) {
        std::cout << importer.GetErrorString() << std::endl;
        return 1;
    }

    test::Model my_model;
    my_model.processNode(scene->mRootNode, scene);

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
