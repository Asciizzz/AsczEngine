#include <Tri3D.cuh>

Tri3D::Tri3D(Vec3D v1, Vec3D v2, Vec3D v3, Color3D color, bool is2sided) {
    this->v1 = v1;
    this->v2 = v2;
    this->v3 = v3;
    this->color = color;
    this->isTwoSided = is2sided;

    this->normal = Vec3D::cross(
        Vec3D::sub(v2, v1),
        Vec3D::sub(v3, v1)
    );
}

Tri3D::Tri3D(Vec3D v1, Vec3D v2, Vec3D v3, Vec3D normal, Color3D color) {
    this->v1 = v1;
    this->v2 = v2;
    this->v3 = v3;
    this->color = color;
    this->normal = normal;
    this->isTwoSided = false;
}


// BETA! read from an obj file and return a list of Tri3D
std::vector<Tri3D> Tri3D::readObj(std::string filename) {
    std::vector<Tri3D> tris;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Could not open file " << filename << std::endl;
        return tris;
    }

    std::vector<Vec3D> verts;
    std::vector<Vec3D> normals;

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string type;
        ss >> type;

        if (type == "v") {
            double x, y, z;
            ss >> x >> y >> z;
            verts.push_back(Vec3D(x, y, z));
        };

        if (type == "vn") {
            double x, y, z;
            ss >> x >> y >> z;
            normals.push_back(Vec3D(x, y, z));
        };

        if (type == "f") {
            /*
            The face format is f v/t/n v/t/n v/t/n

            We will ignore the texture for now
            */

            std::string vtn1, vtn2, vtn3;
            ss >> vtn1 >> vtn2 >> vtn3;

            int v1 = std::stoi(vtn1.substr(0, vtn1.find('/'))) - 1;
            int v2 = std::stoi(vtn2.substr(0, vtn2.find('/'))) - 1;
            int v3 = std::stoi(vtn3.substr(0, vtn3.find('/'))) - 1;

            int n1 = std::stoi(vtn1.substr(vtn1.find_last_of('/') + 1)) - 1;
            int n2 = std::stoi(vtn2.substr(vtn2.find_last_of('/') + 1)) - 1;
            int n3 = std::stoi(vtn3.substr(vtn3.find_last_of('/') + 1)) - 1;

            Vec3D normal = Vec3D::add(
                normals[n1], Vec3D::add(normals[n2], normals[n3])
            );

            Tri3D tri = Tri3D(
                verts[v1], verts[v2], verts[v3], normal, Color3D(255, 255, 255)
            );

            tris.push_back(tri);
        }
    }

    return tris;
}