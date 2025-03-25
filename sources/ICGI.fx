#include "ReShade.fxh"
#include "ReShadeUI.fxh"

namespace IC_GI 
{
uniform int debug_mode <
    ui_category = "Output mode";
    ui_type = "combo";
    ui_label = "Output mode";
    ui_items = "Normal\0"
               "Debug: show Fake GI area light\0"
               "Debug: show Ambient Color\0"
               "Debug: show Estimated Color of Scene\0"
               "Debug: show Normals\0";

    ui_tooltip = "Handy when tuning ambient occlusion settings.";
> = 0;

uniform float gi_strength < __UNIFORM_SLIDER_FLOAT1
    ui_category = "GI Properties";
    ui_min = 0.0; ui_max = 1.0; ui_step = .05;
    ui_label = "Fake GI lighting strength";
    ui_tooltip = "Fake Global Illumination wide-area effect. Every pixel gets some light added from the surrounding area of the image.";
> = 0.6;

uniform float gi_saturation < __UNIFORM_SLIDER_FLOAT1
    ui_category = "GI Properties";
    ui_min = 0.0; ui_max = 1.0; ui_step = .05;
    ui_label = "Fake GI saturation";
    ui_tooltip = "Fake Global Illumination can exaggerate colours in the image too much. Decrease this to reduce the colour saturation of the added light. Increase for more vibrant colours.";
> = 0.6;

uniform float gi_contrast < __UNIFORM_SLIDER_FLOAT1
    ui_category = "GI Properties";
    ui_min = 0; ui_max = 1; ui_step = 0.01;
    ui_tooltip = "Increases contrast relative to average light in surrounding area. \n\nFake Global Illumination can reduce overall contrast; this setting compensates for that and actually improves contrast and clarity compared to the original. \n\nRecommendation: set to roughly half of GI lighting strength.";
    ui_label = "Adaptive contrast enhancement";
> = 0.6;

uniform float OverExposeFactor < __UNIFORM_SLIDER_FLOAT1
    ui_category = "GI Properties";
    ui_min = 0.0; ui_max = 1.0;
    ui_tooltip = "How much the light can directly color edges";
> = 0.5;

uniform bool gi_use_depth <
    ui_category = "GI Properties";	
    ui_label = "Enable Fake GI effects that require depth (below)";    
    ui_tooltip = "If you don't have depth buffer data, or if you don't want the full effect then 2D mode may be faster. \n\nWith depth enabled, it adds big AO, improved local AO with local bounce light, and better direction for lighting.";
> = false;

uniform float bounce_multiplier < __UNIFORM_SLIDER_FLOAT1
    ui_category = "GI Properties";
    ui_min = 0.0; ui_max = 2.0; ui_step = .05;
    ui_label = "Fake GI local bounce strength (requires depth)";
    ui_tooltip = "It uses local depth and colour information to approximate short-range bounced light. \n\nIt only affects areas made darker by ambient occlusion. A bright red pillar next to a white wall will make the wall a bit red, but how red?";
> = 1;

uniform float gi_shape < __UNIFORM_SLIDER_FLOAT1
    ui_category = "GI Properties";
    ui_min = 0; ui_max = .2; ui_step = .01;
    ui_tooltip = "Fake global illumination uses a very blurred version of the image to collect approximate light from a wide area around each pixel. \n\nIf depth is available, it adjusts the offset based on the angle of the local surface. This makes it better pick up colour from facing surfaces, but if set too big it may miss nearby surfaces in favour of further ones. \n\nThe value is the maximum offset, as a fraction of screen size.";
    ui_label = "Fake GI offset";
> = .1;


uniform float EdgeStrength < __UNIFORM_SLIDER_FLOAT1
    ui_category = "Normal Properties";
    ui_min = 0.0; ui_max = 5.0;
    ui_tooltip = "Adjusts the strength of edge detection.";
> = 1.5;

uniform float MinThreshold < __UNIFORM_SLIDER_FLOAT1
    ui_category = "Normal Properties";
    ui_min = 0.0; ui_max = 0.5;
    ui_tooltip = "Minimum value to be considered an edge";
> = .06;

uniform float NormalMapStrength < __UNIFORM_SLIDER_FLOAT1
    ui_category = "Normal Properties";
    ui_min = 0.0; ui_max = 1.0;
    ui_tooltip = "Strength of the normal map effect";
> = 1.0;

uniform float NormalMapRadius < __UNIFORM_SLIDER_FLOAT1
    ui_category = "Normal Properties";
    ui_min = 1.0; ui_max = 20.0;
    ui_tooltip = "Radius of the normal map effect";
> = 10.0;

uniform float BlurRadius < __UNIFORM_SLIDER_FLOAT1
    ui_category = "Normal Properties";
    ui_min = 1.0; ui_max = 200.0;
    ui_tooltip = "How much the GI map is blurred";
> = 15.0;


#if BUFFER_COLOR_BIT_DEPTH == 8
uniform float tone_map < __UNIFORM_SLIDER_FLOAT1
    ui_category = "Advanced Tuning and Configuration";
    ui_min = 1; ui_max = 30; ui_step = .1;
    ui_tooltip = "In the real world we can see a much wider range of brightness than a standard screen can produce. \n\nGames use tone mapping to reduce the dynamic range, especially in bright areas, to fit into display limits. \n\nTo calculate lighting effects like Fake GI accurately on SDR images, we want to undo tone mapping first, then reapply it afterwards. \n\nOptimal value depends on tone mapping method the game uses. You won't find that info published anywhere for most games. \n\nOur compensation is based on Reinhard tonemapping, but hopefully will be close enough if the game uses another curve like ACES. At 5 it's pretty close to ACES in bright areas but never steeper. \n\nApplies for Fake GI in SDR mode only.";
    ui_label = "Tone mapping compensation";
> = 1.5;
#endif

#if BUFFER_COLOR_BIT_DEPTH == 16
uniform bool hdr_nits < 
    ui_category = "Advanced Tuning and Configuration";	
    ui_tooltip = "True if game output is in nits (100 is SDR white); I hear Epic games may use this. False if it's in scRGB (1 is SDR white)";
    ui_label = "HDR FP16 color is in nits.";
> = false;
#endif

float3 undoTonemap(float3 c) {
#if BUFFER_COLOR_BIT_DEPTH > 8
    return c;
#else
    return c / (1.0 - (1.0 - rcp(tone_map)) * c);
#endif
}

float3 reapplyTonemap(float3 c) {
#if BUFFER_COLOR_BIT_DEPTH > 8
    return c;
#else
    return c / ((1 - rcp(tone_map)) * c + 1.0);
#endif
}

float3 toLinear(float3 c) {
    float3 r = c; 
#if BUFFER_COLOR_BIT_DEPTH == 10
    const float m1 = 1305.0 / 8192.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 107.0 / 128.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    float3 powc = pow(max(c, 0), 1.0 / m2);
    r = pow(max(max(powc - c1, 0) / (c2 - c3 * powc), 0), 1.0 / m1);
    r *= 20;
#endif
#if BUFFER_COLOR_BIT_DEPTH == 16
    if (hdr_nits) r *= 0.01;
#endif
    return r;
}

float3 toOutputFormat(float3 c) {
    float3 r = c; 
#if BUFFER_COLOR_BIT_DEPTH == 10
    const float m1 = 1305.0 / 8192.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 107.0 / 128.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    c = c * 0.05;
    float3 powc = pow(max(c, 0), m1);
    r = pow(max((c1 + c2 * powc) / (1 + c3 * powc), 0), m2);
#endif
#if BUFFER_COLOR_BIT_DEPTH == 16
    if (hdr_nits) r *= 100;
#endif
    return r;
}

sampler2D samplerColor {
    Texture = ReShade::BackBufferTex;
#if BUFFER_COLOR_BIT_DEPTH > 8
    SRGBTexture = false;
#else
    SRGBTexture = true;
#endif
};

float4 getBackBufferLinear(float2 texcoord) {	 	
    float4 c = tex2D(samplerColor, texcoord);	
    c.rgb = toLinear(c.rgb);
    return c;
}

sampler2D samplerDepth {
    Texture = ReShade::DepthBufferTex;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
};

float getDepth(float2 texcoord) {
#if RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN
    texcoord.y = 1.0 - texcoord.y;
#endif
    texcoord.x /= RESHADE_DEPTH_INPUT_X_SCALE;
    texcoord.y /= RESHADE_DEPTH_INPUT_Y_SCALE;
#if RESHADE_DEPTH_INPUT_X_PIXEL_OFFSET
    texcoord.x -= RESHADE_DEPTH_INPUT_X_PIXEL_OFFSET * BUFFER_RCP_WIDTH;
#else
    texcoord.x -= RESHADE_DEPTH_INPUT_X_OFFSET / 2.000000001;
#endif
#if RESHADE_DEPTH_INPUT_Y_PIXEL_OFFSET
    texcoord.y += RESHADE_DEPTH_INPUT_Y_PIXEL_OFFSET * BUFFER_RCP_HEIGHT;
#else
    texcoord.y += RESHADE_DEPTH_INPUT_Y_OFFSET / 2.000000001;
#endif
    float depth = (float)tex2D(samplerDepth, texcoord);
    depth *= RESHADE_DEPTH_MULTIPLIER;
#if RESHADE_DEPTH_INPUT_IS_LOGARITHMIC
    const float C = 0.01;
    depth = (exp(depth * log(C + 1.0)) - 1.0) / C;
#endif
#if RESHADE_DEPTH_INPUT_IS_REVERSED
    depth = 1 - depth;
#endif
    const float N = 1.0;
    depth /= RESHADE_DEPTH_LINEARIZATION_FAR_PLANE - depth * (RESHADE_DEPTH_LINEARIZATION_FAR_PLANE - N);
    return depth;
}

#ifndef FAKE_GI_WIDTH
    #define FAKE_GI_WIDTH 192
#endif
#ifndef FAKE_GI_HEIGHT
    #define FAKE_GI_HEIGHT 108
#endif

texture GITexture {
    Width = FAKE_GI_WIDTH * 2;
    Height = FAKE_GI_HEIGHT * 2;
    Format = RGBA16F;
    MipLevels = 4;
};

sampler GITextureSampler {
    Texture = GITexture;
};

texture BlurTex {
    Width = FAKE_GI_WIDTH;
    Height = FAKE_GI_HEIGHT;
    Format = RGBA16F;
};

sampler BlurSample {
    Texture = BlurTex;
};

// Helper functions to reduce duplication:
float grayscale(float3 c) {
    return dot(c, float3(0.3333, 0.3333, 0.3333));
}
float edgeValue(float dx, float dy) {
    float val = saturate(EdgeStrength * sqrt(dx * dx + dy * dy));
    return (val <= MinThreshold) ? 0.0 : val;
}

float4 startGI_PS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : COLOR {
    float4 c = getBackBufferLinear(texcoord);
    c.rgb = undoTonemap(c.rgb);
    if (gi_use_depth) {
        float depth = getDepth(texcoord);					
        c.w = depth;
    }
    return c;
}

float4 bigBlur(sampler s, float2 texcoord, float4 steps) {
    float2 pixelsize = 1.0 / float2(FAKE_GI_WIDTH, FAKE_GI_HEIGHT);
    float4 sum = 0.0;
    float weightSum = 0.0;
    int radius = (int)BlurRadius;
    float sigma = radius / 2.0;
    float twoSigmaSq = 2.0 * sigma * sigma;
    float invSigmaRoot = 1.0 / sqrt(3.14159265 * twoSigmaSq);

    [loop]
    for (int x = -radius; x <= radius; x++) {
        [loop]
        for (int y = -radius; y <= radius; y++) {
            float2 offset = float2(x, y) * pixelsize;
            float weight = invSigmaRoot * exp(-(x * x + y * y) / twoSigmaSq);
            sum += tex2D(s, texcoord + offset) * weight;
            weightSum += weight;
        }
    }
    return sum / weightSum;
}

float4 bigBlur_PS(float4 pos : SV_Position, float2 texcoord : TEXCOORD) : COLOR {
    return bigBlur(GITextureSampler, texcoord, float4(10.5, 1.5, 3.5, 0.5));
}

// Normal estimation effect for when there is no depth buffer present.
// This gives less-than-ideal results but is significantly better than nothing.
// This also has the benefit of being usable in 2D games or games where the depth buffer is not accessible.

float4 getNormalPosition(float4 pos, float2 texcoord)
{
    // Get the color at the center pixel
    float3 centerColor = tex2D(ReShade::BackBuffer, texcoord).rgb;

    // Calculate pixel size
    float2 pixelSize = BUFFER_PIXEL_SIZE;

    // Get the colors of the neighboring pixels
    float3 leftColor   = tex2D(ReShade::BackBuffer, texcoord + float2(-1, 0) * pixelSize).rgb;
    float3 rightColor  = tex2D(ReShade::BackBuffer, texcoord + float2( 1, 0) * pixelSize).rgb;
    float3 upColor     = tex2D(ReShade::BackBuffer, texcoord + float2( 0, 1) * pixelSize).rgb;
    float3 downColor   = tex2D(ReShade::BackBuffer, texcoord + float2( 0,-1) * pixelSize).rgb;

    // Convert colors to grayscale
    float centerGray = dot(centerColor, float3(0.3333, 0.3333, 0.3333));
    float leftGray   = dot(leftColor,   float3(0.3333, 0.3333, 0.3333));
    float rightGray  = dot(rightColor,  float3(0.3333, 0.3333, 0.3333));
    float upGray     = dot(upColor,     float3(0.3333, 0.3333, 0.3333));
    float downGray   = dot(downColor,   float3(0.3333, 0.3333, 0.3333));

    // Calculate gradients
    float gradientX = 0.5 * (rightGray - leftGray);
    float gradientY = 0.5 * (upGray - downGray);

    // Calculate edge value
    float edgeValue = saturate(EdgeStrength * sqrt(gradientX * gradientY + gradientY * gradientY));
    if (edgeValue <= MinThreshold) edgeValue = 0.0;

    // Initialize normal and gradient variables
    const float3 flatNormal = float3(0.0, 0.0, 1.0);
    float2 gradients = 0.0;
    float weight = 0.0;

    // Calculate step size
    float step = max(1.0, NormalMapRadius * 0.2);

    // Loop through neighboring pixels to calculate gradients
    [loop]
    for (float x = -NormalMapRadius; x <= NormalMapRadius; x += step)
    {
        for (float y = -NormalMapRadius; y <= NormalMapRadius; y += step)
        {
            float2 offset = float2(x, y) * pixelSize;
            float3 sampleColor = tex2D(ReShade::BackBuffer, texcoord + offset).rgb;

            float sampleGray      = dot(sampleColor, float3(0.3333, 0.3333, 0.3333));
            float sampleLeftGray  = dot(tex2D(ReShade::BackBuffer, texcoord + offset + float2(-1, 0) * pixelSize).rgb, float3(0.3333, 0.3333, 0.3333));
            float sampleRightGray = dot(tex2D(ReShade::BackBuffer, texcoord + offset + float2( 1, 0) * pixelSize).rgb, float3(0.3333, 0.3333, 0.3333));
            float sampleUpGray    = dot(tex2D(ReShade::BackBuffer, texcoord + offset + float2( 0, 1) * pixelSize).rgb, float3(0.3333, 0.3333, 0.3333));
            float sampleDownGray  = dot(tex2D(ReShade::BackBuffer, texcoord + offset + float2( 0,-1) * pixelSize).rgb, float3(0.3333, 0.3333, 0.3333));

            float sampleGradientX = 0.5 * (sampleRightGray - sampleLeftGray);
            float sampleGradientY = 0.5 * (sampleUpGray - sampleDownGray);
            float localEdgeValue = saturate(EdgeStrength * sqrt(sampleGradientX * sampleGradientX + sampleGradientY * sampleGradientY));
            if (localEdgeValue <= MinThreshold) localEdgeValue = 0.0;

            float distance = length(float2(x, y));
            float distanceWeight = 1.0 / (1.0 + distance * 0.5);
            localEdgeValue *= distanceWeight;

            gradients += localEdgeValue * float2(x, y);
            weight += localEdgeValue;
        }
    }

    // Normalize gradients
    if (weight > 0.0)
        gradients /= weight;
    gradients /= NormalMapStrength;

    // Calculate falloff and normal
    float falloff = saturate(length(gradients) / NormalMapRadius);
    float3 normal = normalize(float3(-gradients.x, -gradients.y, 1.0));
    normal = lerp(normal, flatNormal, falloff);

    return float4(normal * -1.0, 1.0);
}

float3 IC_GI_PS(float4 screenPos : SV_Position, float2 uv : TexCoord) : SV_Target
{
    // Read initial color and depth (if enabled)
    float3 mainColor = getBackBufferLinear(uv).rgb;
    float depthValue = 0;
    if (gi_use_depth)
    {
        depthValue = getDepth(uv);
    }

    // Convert to a pre-tonemapped space
    float3 smoothColor = mainColor;
    mainColor = undoTonemap(mainColor);
    smoothColor = undoTonemap(smoothColor);

    // Create a rough "bounce" term
    float3 bounceColor = smoothColor * normalize(mainColor);

    // Sample a region for bounce lighting from the GI texture
    float4 bounceArea = tex2Dlod(GITextureSampler, float4(uv.x, uv.y, 0, 2.5));

    // Calculate offset vector based on depth or estimated normals
    float2 giOffset = 0;
    if (gi_use_depth)
    {
        float4 localSlope = float4(ddx(depthValue), ddy(depthValue), 0.1 * BUFFER_PIXEL_SIZE);
        giOffset = normalize(localSlope).xy * gi_shape;
    }
    else
    {
        float3 normalEst = getNormalPosition(screenPos, uv).xyz;
        giOffset = normalEst.xy * gi_shape;
    }
    float2 normal = giOffset / gi_shape;

    // Get blurred GI color
    float4 giSample = tex2D(BlurSample, uv + giOffset) + float4(0.001, 0.001, 0.001, 0);
    float giBrightness = max(giSample.r, max(giSample.g, giSample.b)) + min(giSample.r, min(giSample.g, giSample.b));
    giBrightness *= giBrightness;

    // Compute indirect lighting contributions
    float lightFactor = giBrightness + 0.005;
    float3 unlitColor2 = mainColor / lightFactor;
    bounceColor = lerp(bounceColor, unlitColor2 * max(0, 2 * bounceArea.rgb - mainColor), bounce_multiplier);

    float smoothBrightness = max(smoothColor.r, max(smoothColor.g, smoothColor.b))
                           + min(smoothColor.r, min(smoothColor.g, smoothColor.b));
    float ambientTerm = lerp(giBrightness, smoothBrightness, 0.5);
    float3 unlitColor = mainColor / ambientTerm;

    // Desaturate GI if desired
    float3 desatGI = lerp(
        (float3)(length(giSample.rgb) / sqrt(2)),
        giSample.rgb,
        length(giSample.rgb * gi_saturation)
            / length(mainColor * (1 - gi_saturation) + giSample.rgb * gi_saturation)
    );
    float3 giBounce = unlitColor * 2 * desatGI;

    // Compute adaptive contrast
    float currContrast = max(
        sqrt(2 * length(mainColor) / length(bounceArea.rgb + giSample.rgb)), 0.75
    );
    float adjustedLength = clamp(
        length(min((giSample.rgb + mainColor + mainColor - smoothColor) / 2, giBounce)),
        length(mainColor),
        1.5 * length(mainColor)
    );

    // Apply bounce lighting
    // Step A: Prepare mainColor from GI bounce
    float3 originalColor = mainColor;
    mainColor = normalize(giBounce) * adjustedLength;
    mainColor *= lerp(1, currContrast, gi_contrast);

    // Step B: Emphasize GI highlights
    float3 intermediateColor = mainColor;
    float4 originalGI = giSample;

    // Boost GI brightness
    giSample *= float4(giBrightness, giBrightness, giBrightness, 1);
    float3 blowoutColor = giSample.xyz * giSample.xyz;
    float blowoutIntensity = saturate(dot(blowoutColor, float3(0.3333, 0.3333, 0.3333)));

    // Blend GI highlight
    giSample = lerp(giSample, originalGI, 0.5);

    // Step C: Blend GI highlight into mainColor
    mainColor *= (giSample.xyz + 0.75);
    mainColor = lerp(mainColor, intermediateColor, 0.5);

    // Step D: Over-exposure blend
    float3 blownColor = lerp(mainColor, giSample.xyz, blowoutIntensity);
    mainColor = lerp(mainColor, blownColor, OverExposeFactor * 0.15);
    mainColor = lerp(originalColor, mainColor, gi_strength);

    // Step E: Reapply tone mapping
    mainColor = reapplyTonemap(mainColor);

    // Step F: Debug modes
    if (debug_mode == 1) mainColor = giSample.rgb;
    if (debug_mode == 2) mainColor = ambientTerm;
    if (debug_mode == 3) mainColor = unlitColor;
    if (debug_mode == 4) mainColor = float3((-normal +1) * 0.5 ,1);

    // Step G: Final output
    mainColor = toOutputFormat(mainColor);
    return mainColor;
}

void PostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD) {
    texcoord.x = (id == 2) ? 2.0 : 0.0;
    texcoord.y = (id == 1) ? 2.0 : 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}


technique IC_GI <
    ui_tooltip = "Fake Global Illumination using a wide-area effect. It can be used in 2D mode, or with depth data for improved local bounce light and better direction for lighting.";
> {	
    pass makeGI {
        VertexShader = PostProcessVS;
        PixelShader = startGI_PS;
        RenderTarget = GITexture;		
    }	
    pass {
        VertexShader = PostProcessVS;
        PixelShader = bigBlur_PS;
        RenderTarget = BlurTex;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = IC_GI_PS;
#if BUFFER_COLOR_BIT_DEPTH > 8
        SRGBWriteEnable = false;
#else
        SRGBWriteEnable = true;
#endif
    }	
}
}

/*
MIT License

Copyright (c) 2021 IC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
*/