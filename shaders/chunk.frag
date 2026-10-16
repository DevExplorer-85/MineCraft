#version 330 core

in vec3 voxel_color;
in vec2 uv;
in float shading;
in vec3 world_pos;

flat in int face_id;
flat in int voxel_id;

layout(location = 0) out vec4 fragColor;

const vec3 gamma = vec3(2.2);
const vec3 inv_gamma = 1.0 / gamma;

const vec3 normals[6] = vec3[6](
    vec3(0, 1, 0), vec3(0, -1, 0),
    vec3(1, 0, 0), vec3(-1, 0, 0),
    vec3(0, 0, -1), vec3(0, 0, 1)
);

const float CAVE_MIN_LIGHT = 0.05;
const float CAVE_FADE_DEPTH = 14.0;

uniform sampler2DArray u_texture_array_0;
uniform sampler2D u_heightmap;
uniform float u_underwater;
uniform vec3 u_cam_pos;
uniform float u_time;
uniform int u_pass;

const vec3 UNDERWATER_FOG = vec3(0.05, 0.25, 0.45);

const vec3 FOG_COLOR = vec3(0.58, 0.78, 0.95);   // BG_COLOR ke barabar
const float FOG_START = 120.0;
const float FOG_END   = 520.0;

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float water_height(vec2 p, float t)
{
    float n1 = vnoise(p * 0.35 + vec2(t * 0.20, t * 0.12));
    float n2 = vnoise(p * 0.80 + vec2(-t * 0.28, t * 0.18));
    float n3 = vnoise(p * 1.70 + vec2(t * 0.10, -t * 0.35));
    return n1 * 0.55 + n2 * 0.30 + n3 * 0.15;
}

float cave_light()
{
    vec3 p = world_pos + normals[face_id] * 0.5;
    ivec3 a = ivec3(floor(p));

    ivec2 size = textureSize(u_heightmap, 0);
    if (a.x < 0 || a.z < 0 || a.x >= size.x || a.z >= size.y)
        return 1.0;

    int h = int(texelFetch(u_heightmap, ivec2(a.x, a.z), 0).r * 255.0 + 0.5);
    int depth = h - a.y;

    if (depth <= 0)
        return 1.0;

    float t = smoothstep(0.0, CAVE_FADE_DEPTH, float(depth));
    return mix(1.0, CAVE_MIN_LIGHT, t);
}

void main()
{
    if (u_pass == 0 && voxel_id == 8) discard;
    if (u_pass == 1 && voxel_id != 8) discard;

    vec2 face_uv = uv;
    face_uv.x = uv.x / 3.0 - min(face_id, 2) / 3.0;

    // naye leaf IDs (9 oak, 10 spruce) leaves texture (layer 6) use karte hain
       int tex_id = voxel_id;
    if (voxel_id == 9 || voxel_id == 10) tex_id = 6;     // oak / spruce leaves
    if (voxel_id == 11) tex_id = 7;                      // planks = wood texture
    vec3 tex_col = texture(u_texture_array_0, vec3(face_uv, tex_id)).rgb;
    if (voxel_id == 11) tex_col = clamp(tex_col * vec3(1.55, 1.3, 0.95), 0.0, 1.0);   // halka lakdi rang
    if (voxel_id == 8) {   // WATER (flowing, procedural)
        vec2 p = world_pos.xz;

        float e = 0.15;
        float h  = water_height(p, u_time);
        float hx = water_height(p + vec2(e, 0.0), u_time);
        float hz = water_height(p + vec2(0.0, e), u_time);
        vec3 n = normalize(vec3((h - hx) * 3.0, 1.0, (h - hz) * 3.0));

        vec3 view_dir = normalize(u_cam_pos - world_pos);
        float fres = pow(1.0 - clamp(dot(n, view_dir), 0.0, 1.0), 3.0);
        vec3 light_dir = normalize(vec3(0.4, 0.8, 0.3));
        float spec = pow(clamp(dot(reflect(-light_dir, n), view_dir), 0.0, 1.0), 40.0);

        vec3 deep    = vec3(0.05, 0.26, 0.52);
        vec3 shallow = vec3(0.16, 0.48, 0.74);
        tex_col = mix(deep, shallow, smoothstep(0.30, 0.70, h));
        tex_col = mix(tex_col, vec3(0.62, 0.80, 0.95), fres * 0.35);
        tex_col += vec3(spec) * 0.35;
    }

    tex_col = pow(tex_col, gamma);

    if (voxel_id == 6) tex_col *= 1.25;                      // cherry leaves
    if (voxel_id == 9 || voxel_id == 10) tex_col *= 1.1;     // oak / spruce leaves

    if (voxel_id == 8)
        tex_col *= 0.9 + 0.1 * shading;
    else
        tex_col *= shading * cave_light();

    tex_col = pow(tex_col, inv_gamma);

    if (u_underwater > 0.5) {
        float d = length(world_pos - u_cam_pos);

        float caustic = 0.85 + 0.15 * sin(world_pos.x * 1.7 + u_time * 1.5)
                                    * sin(world_pos.z * 1.9 - u_time * 1.2);

        tex_col *= vec3(0.70, 0.90, 1.0) * caustic;
        float fog = 1.0 - exp(-d * 0.03);
        tex_col = mix(tex_col, UNDERWATER_FOG, clamp(fog, 0.0, 1.0));
    }
    else {
        float d = length(world_pos - u_cam_pos);

        float fog = smoothstep(FOG_START, FOG_END, d);
        fog = fog * fog * 0.95;

        float low = 1.0 - smoothstep(14.0, 55.0, world_pos.y);
        fog = clamp(fog + low * smoothstep(20.0, 200.0, d) * 0.25, 0.0, 1.0);

        tex_col = mix(tex_col, FOG_COLOR, fog);
    }

    float alpha = 1.0;
    if (voxel_id == 8) {
        float d = length(world_pos - u_cam_pos);
        alpha = clamp(0.45 + d * 0.004, 0.45, 0.75);
    }
    fragColor = vec4(tex_col, alpha);
}