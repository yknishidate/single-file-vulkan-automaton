
#include <set>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

using vkBU = vk::BufferUsageFlagBits;
using vkIU = vk::ImageUsageFlagBits;
using vkMP = vk::MemoryPropertyFlagBits;
using vkDT = vk::DescriptorType;
using vkSS = vk::ShaderStageFlagBits;

// ----------------------------------------------------------------------------------------------------------
// Globals
// ----------------------------------------------------------------------------------------------------------
constexpr int WIDTH = 1024;
constexpr int HEIGHT = 1024;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
std::vector<const char*> validationLayers;
//#ifdef _DEBUG
constexpr bool enableValidationLayers = true;
//#else
//constexpr bool enableValidationLayers = false;
//#endif

// ----------------------------------------------------------------------------------------------------------
// Functuins
// ----------------------------------------------------------------------------------------------------------
VKAPI_ATTR VkBool32 VKAPI_CALL
debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes, VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData, void*)
{
    std::cerr << "messageIndexName   = " << pCallbackData->pMessageIdName << "\n";
    for (uint8_t i = 0; i < pCallbackData->objectCount; i++) {
        vk::ObjectType type = static_cast<vk::ObjectType>(pCallbackData->pObjects[i].objectType);
        std::cerr << "objectType      = " << vk::to_string(type) << "\n";
    }
    std::cerr << pCallbackData->pMessage << "\n\n";
    return VK_FALSE;
}

void transitionImageLayout(vk::CommandBuffer cmdBuf, vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eAllCommands;
    vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eAllCommands;

    vk::ImageMemoryBarrier barrier{};
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(image);
    barrier.setOldLayout(oldLayout);
    barrier.setNewLayout(newLayout);
    barrier.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

    using vkAF = vk::AccessFlagBits;
    switch (oldLayout) {
        case vk::ImageLayout::eUndefined:
            barrier.srcAccessMask = {};
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            barrier.srcAccessMask = vkAF::eTransferRead;
            break;
        case vk::ImageLayout::eTransferDstOptimal:
            barrier.srcAccessMask = vkAF::eTransferWrite;
            break;
        default:
            break;
    }
    switch (newLayout) {
        case vk::ImageLayout::eTransferDstOptimal:
            barrier.dstAccessMask = vkAF::eTransferWrite;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            barrier.dstAccessMask = vkAF::eTransferRead;
            break;
        default:
            break;
    }
    cmdBuf.pipelineBarrier(srcStageMask, dstStageMask, {}, {}, {}, barrier);
}

uint32_t findMemoryType(const vk::PhysicalDevice physicalDevice, const uint32_t typeFilter, const vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i))
            && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type");
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats)
{
    if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
        return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
    }
    for (const auto& format : formats) {
        if (format.format == vk::Format::eB8G8R8A8Unorm && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }
    throw std::runtime_error("found no suitable surface format");
}

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eFifoRelaxed) {
            return availablePresentMode;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}


// ----------------------------------------------------------------------------------------------------------
// Structs
// ----------------------------------------------------------------------------------------------------------
struct Buffer
{
    vk::Device device;
    vk::PhysicalDevice physicalDevice;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DeviceSize size;
    vk::BufferUsageFlags usage;
    uint64_t deviceAddress;
    void* mapped = nullptr;

    void create(vk::Device device, vk::DeviceSize size, vk::BufferUsageFlags usage)
    {
        this->device = device;
        this->size = size;
        this->usage = usage;
        buffer = device.createBufferUnique({ {}, size, usage });
    }

    void bindMemory(vk::PhysicalDevice physicalDevice, vk::MemoryPropertyFlags properties)
    {
        vk::MemoryRequirements requirements = device.getBufferMemoryRequirements(*buffer);
        uint32_t memoryTypeIndex = findMemoryType(physicalDevice, requirements.memoryTypeBits, properties);
        vk::MemoryAllocateInfo allocInfo{ requirements.size, memoryTypeIndex };

        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            vk::MemoryAllocateFlagsInfo flagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };
            allocInfo.pNext = &flagsInfo;

            memory = device.allocateMemoryUnique(allocInfo);
            device.bindBufferMemory(*buffer, *memory, 0);

            vk::BufferDeviceAddressInfoKHR bufferDeviceAI{ *buffer };
            deviceAddress = device.getBufferAddressKHR(&bufferDeviceAI);
        } else {
            memory = device.allocateMemoryUnique(allocInfo);
            device.bindBufferMemory(*buffer, *memory, 0);
        }
    }

    void fillData(void* data)
    {
        if (!mapped) {
            mapped = device.mapMemory(*memory, 0, size);
        }
        memcpy(mapped, data, static_cast<size_t>(size));
    }

    vk::DescriptorBufferInfo createDescInfo()
    {
        return vk::DescriptorBufferInfo{ *buffer, 0, size };
    }
};

struct Image
{
    vk::Device device;
    vk::UniqueImage image;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory memory;
    vk::Extent2D extent;
    vk::Format format;
    vk::ImageLayout imageLayout;

    void create(vk::Device device, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage)
    {
        this->device = device;
        this->extent = extent;
        this->format = format;

        image = device.createImageUnique(
            vk::ImageCreateInfo{}
            .setImageType(vk::ImageType::e2D)
            .setExtent({ extent.width, extent.height, 1 })
            .setMipLevels(1)
            .setArrayLayers(1)
            .setFormat(format)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(usage));
    }

    void bindMemory(vk::PhysicalDevice physicalDevice)
    {
        vk::MemoryRequirements requirements = device.getImageMemoryRequirements(*image);
        uint32_t memoryTypeIndex = findMemoryType(physicalDevice, requirements.memoryTypeBits,
                                                  vk::MemoryPropertyFlagBits::eDeviceLocal);
        memory = device.allocateMemoryUnique({ requirements.size, memoryTypeIndex });
        device.bindImageMemory(*image, *memory, 0);
    }

    void createImageView()
    {
        view = device.createImageViewUnique(
            vk::ImageViewCreateInfo{}
            .setImage(*image)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(format)
            .setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }));
    }

    vk::DescriptorImageInfo createDescInfo()
    {
        return vk::DescriptorImageInfo{ {}, *view, imageLayout };
    }
};

struct UniformData
{
    int frame = 0;
};

// ----------------------------------------------------------------------------------------------------------
// Application
// ----------------------------------------------------------------------------------------------------------
class Application
{
public:

    ~Application()
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            device->destroyFence(inFlightFences[i]);
        }
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
    }

private:

    GLFWwindow* window;
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::UniqueSurfaceKHR surface;

    vk::UniqueDevice device;
    vk::PhysicalDevice physicalDevice;

    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> computeCommandBuffers;

    uint32_t computeFamily;
    uint32_t presentFamily;
    vk::Queue computeQueue;
    vk::Queue presentQueue;

    vk::UniqueSwapchainKHR swapChain;
    vk::PresentModeKHR presentMode;
    vk::Format format;
    vk::Extent2D extent;
    std::vector<vk::Image> swapChainImages;

    Image inputImage;
    Image outputImage;

    vk::UniqueShaderModule shaderModule;
    vk::PipelineShaderStageCreateInfo shaderStage;

    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueDescriptorSetLayout descSetLayout;

    vk::UniqueDescriptorPool descPool;
    vk::UniqueDescriptorSet descSet;

    std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    size_t currentFrame = 0;

    UniformData uniformData;
    Buffer uniformBuffer;

    const std::vector<const char*> requiredExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_KHR_MAINTENANCE3_EXTENSION_NAME,
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    };

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "VulkanAutomaton", nullptr, nullptr);
    }

    void initVulkan()
    {
        createInstance();
        std::cout << "created instance\n";
        createSurface();
        std::cout << "created surface\n";
        createDevice();
        std::cout << "created device\n";
        createSwapChain();
        std::cout << "created swap chain\n";
        inputImage = createStorageImage();
        outputImage = createStorageImage();
        std::cout << "created storage image\n";
        createUniformBuffer();
        std::cout << "created uniform buffer\n";
        loadShaders();
        std::cout << "loaded shaders\n";
        createComputePipeLine();
        std::cout << "created pipeline\n";
        createDescriptorSets();
        std::cout << "created desc sets\n";
        buildCommandBuffers();
        std::cout << "built command buffers\n";
        createSyncObjects();
        std::cout << "create sync objs\n";
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
            updateUniformBuffer();
            if (uniformData.frame % 10 == 0) {
                std::cout << "frame: " << uniformData.frame << std::endl;
            }
        }
        device->waitIdle();
    }

    void createInstance()
    {
        // Get GLFW extensions
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // Setup DynamicLoader (see https://github.com/KhronosGroup/Vulkan-Hpp)
        static vk::DynamicLoader dl;
        auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        if (enableValidationLayers) {
            validationLayers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        // Create instance
        vk::ApplicationInfo appInfo;
        appInfo.setPApplicationName("VulkanPathtracing");
        appInfo.setApiVersion(VK_API_VERSION_1_2);
        instance = vk::createInstanceUnique({ {}, &appInfo, validationLayers, extensions });
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

        physicalDevice = instance->enumeratePhysicalDevices().front();
        if (enableValidationLayers) {
            createDebugMessenger();
        }
    }

    void createDebugMessenger()
    {
        using vkDUMS = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        using vkDUMT = vk::DebugUtilsMessageTypeFlagBitsEXT;
        messenger = instance->createDebugUtilsMessengerEXTUnique(
            vk::DebugUtilsMessengerCreateInfoEXT{}
            .setMessageSeverity(vkDUMS::eWarning | vkDUMS::eError)
            .setMessageType(vkDUMT::eGeneral | vkDUMT::ePerformance | vkDUMT::eValidation)
            .setPfnUserCallback(&debugUtilsMessengerCallback));
    }

    void createSurface()
    {
        VkSurfaceKHR _surface;
        VkResult res = glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        vk::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE> _deleter(*instance);
        surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), _deleter);
    }

    void createDevice()
    {
        findQueueFamilies();
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { computeFamily, presentFamily };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo{ {}, queueFamily, 1, &queuePriority };
            queueCreateInfos.push_back(queueCreateInfo);
        }

        vk::DeviceCreateInfo createInfo{ {}, queueCreateInfos, validationLayers, requiredExtensions };
        device = physicalDevice.createDeviceUnique(createInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

        computeQueue = device->getQueue(computeFamily, 0);
        presentQueue = device->getQueue(presentFamily, 0);
        commandPool = device->createCommandPoolUnique(
            vk::CommandPoolCreateInfo{}
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
            .setQueueFamilyIndex(computeFamily));
    }

    void findQueueFamilies()
    {
        int i = 0;
        for (const auto& queueFamily : physicalDevice.getQueueFamilyProperties()) {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) {
                computeFamily = i;
            }
            vk::Bool32 presentSupport = physicalDevice.getSurfaceSupportKHR(i, *surface);
            if (presentSupport) {
                presentFamily = i;
            }
            if (computeFamily != -1 && presentFamily != -1) {
                break;
            }
            i++;
        }
    }

    void createSwapChain()
    {
        // Query swapchain support
        auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        auto formats = physicalDevice.getSurfaceFormatsKHR(*surface);
        auto presentModes = physicalDevice.getSurfacePresentModesKHR(*surface);

        // Choose swapchain settings
        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(formats);
        format = surfaceFormat.format;
        presentMode = chooseSwapPresentMode(presentModes);
        extent = vk::Extent2D{ WIDTH, HEIGHT };
        uint32_t imageCount = capabilities.minImageCount + 1;

        // Create swap chain
        vk::SwapchainCreateInfoKHR createInfo{};
        createInfo.setSurface(*surface);
        createInfo.setMinImageCount(imageCount);
        createInfo.setImageFormat(format);
        createInfo.setImageColorSpace(surfaceFormat.colorSpace);
        createInfo.setImageExtent(extent);
        createInfo.setImageArrayLayers(1);
        createInfo.setImageUsage(vkIU::eTransferDst);
        createInfo.setPreTransform(capabilities.currentTransform);
        createInfo.setPresentMode(presentMode);
        createInfo.setClipped(true);
        if (computeFamily != presentFamily) {
            std::vector<uint32_t> queueFamilyIndices{ computeFamily, presentFamily };
            createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
            createInfo.setQueueFamilyIndices(queueFamilyIndices);
        }
        swapChain = device->createSwapchainKHRUnique(createInfo);
        swapChainImages = device->getSwapchainImagesKHR(*swapChain);
    }

    Image createStorageImage()
    {
        // TODO: support computeFamily != presentFamily
        Image image;
        image.create(*device, extent, format, vkIU::eStorage | vkIU::eTransferSrc | vkIU::eTransferDst);
        image.bindMemory(physicalDevice);
        image.createImageView();

        // Set image layout
        image.imageLayout = vk::ImageLayout::eGeneral;
        vk::UniqueCommandBuffer cmdBuf = createCommandBuffer();
        transitionImageLayout(*cmdBuf, *image.image, vk::ImageLayout::eUndefined, image.imageLayout);
        submitCommandBuffer(*cmdBuf);
        return std::move(image);
    }

    vk::UniqueCommandBuffer createCommandBuffer()
    {
        vk::CommandBufferAllocateInfo allocInfo{ *commandPool, vk::CommandBufferLevel::ePrimary, 1 };
        vk::UniqueCommandBuffer cmdBuf = std::move(device->allocateCommandBuffersUnique(allocInfo).front());
        cmdBuf->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        return cmdBuf;
    }

    void submitCommandBuffer(vk::CommandBuffer& cmdBuf)
    {
        cmdBuf.end();
        vk::UniqueFence fence = device->createFenceUnique({});
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(cmdBuf);
        computeQueue.submit(submitInfo, *fence);
        vk::Result res = device->waitForFences(*fence, true, UINT64_MAX);
        assert(res == vk::Result::eSuccess);
    }

    void createUniformBuffer()
    {
        uniformBuffer.create(*device, sizeof(UniformData), vkBU::eUniformBuffer);
        uniformBuffer.bindMemory(physicalDevice, vkMP::eHostVisible | vkMP::eHostCoherent);
        updateUniformBuffer();
    }

    void updateUniformBuffer()
    {
        uniformData.frame += 1;
        uniformBuffer.fillData(&uniformData);
    }

    void loadShaders()
    {
        shaderModule = createShaderModule("shaders/compute.comp.spv");
        shaderStage = { {}, vkSS::eCompute, *shaderModule, "main" };
    }

    vk::UniqueShaderModule createShaderModule(const std::string& filename)
    {
        const std::vector<char> code = readFile(filename);
        return device->createShaderModuleUnique({ {}, code.size(), reinterpret_cast<const uint32_t*>(code.data()) });
    }

    void createComputePipeLine()
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        bindings.push_back({ 0, vkDT::eStorageImage, 1, vkSS::eCompute });
        bindings.push_back({ 1, vkDT::eStorageImage, 1, vkSS::eCompute });
        bindings.push_back({ 2, vkDT::eUniformBuffer, 1, vkSS::eCompute });

        descSetLayout = device->createDescriptorSetLayoutUnique({ {}, bindings });
        pipelineLayout = device->createPipelineLayoutUnique({ {}, *descSetLayout });

        // Create pipeline
        vk::ComputePipelineCreateInfo createInfo;
        createInfo.setStage(shaderStage);
        createInfo.setLayout(*pipelineLayout);
        auto res = device->createComputePipelineUnique(nullptr, createInfo);
        if (res.result == vk::Result::eSuccess) {
            pipeline = std::move(res.value);
        } else {
            throw std::runtime_error("failed to create compute pipeline.");
        }
    }

    void createDescriptorSets()
    {
        createDescPool();
        descSet = std::move(device->allocateDescriptorSetsUnique({ *descPool, *descSetLayout }).front());
        updateDescSet();
    }

    void createDescPool()
    {
        std::vector<vk::DescriptorPoolSize> poolSizes{ {vkDT::eStorageImage, 2}, {vkDT::eUniformBuffer, 1} };

        descPool = device->createDescriptorPoolUnique(
            vk::DescriptorPoolCreateInfo{}
            .setPoolSizes(poolSizes)
            .setMaxSets(1)
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet));
    }

    void updateDescSet()
    {
        std::vector<vk::WriteDescriptorSet> writeDescSets;
        vk::DescriptorImageInfo inputImageInfo = inputImage.createDescInfo();
        vk::DescriptorImageInfo outputImageInfo = outputImage.createDescInfo();
        vk::DescriptorBufferInfo uniformBufferInfo = uniformBuffer.createDescInfo();
        vk::WriteDescriptorSet inputImageWrite = createImageWrite(inputImageInfo, vkDT::eStorageImage, 0);
        vk::WriteDescriptorSet outputImageWrite = createImageWrite(outputImageInfo, vkDT::eStorageImage, 1);
        vk::WriteDescriptorSet uniformBufferWrite = createBufferWrite(uniformBufferInfo, vkDT::eUniformBuffer, 2);
        writeDescSets.push_back(inputImageWrite);
        writeDescSets.push_back(outputImageWrite);
        writeDescSets.push_back(uniformBufferWrite);
        device->updateDescriptorSets(writeDescSets, nullptr);
    }

    vk::WriteDescriptorSet createImageWrite(vk::DescriptorImageInfo imageInfo, vk::DescriptorType type, uint32_t binding)
    {
        vk::WriteDescriptorSet imageWrite{};
        imageWrite.setDstSet(*descSet);
        imageWrite.setDescriptorType(type);
        imageWrite.setDescriptorCount(1);
        imageWrite.setDstBinding(binding);
        imageWrite.setImageInfo(imageInfo);
        return imageWrite;
    }

    vk::WriteDescriptorSet createBufferWrite(vk::DescriptorBufferInfo bufferInfo, vk::DescriptorType type, uint32_t binding)
    {
        vk::WriteDescriptorSet bufferWrite{};
        bufferWrite.setDstSet(*descSet);
        bufferWrite.setDescriptorType(type);
        bufferWrite.setDescriptorCount(1);
        bufferWrite.setDstBinding(binding);
        bufferWrite.setBufferInfo(bufferInfo);
        return bufferWrite;
    }

    void buildCommandBuffers()
    {
        allocateDrawCommandBuffers();
        for (int32_t i = 0; i < computeCommandBuffers.size(); ++i) {
            computeCommandBuffers[i]->begin(vk::CommandBufferBeginInfo{});
            computeCommandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
            computeCommandBuffers[i]->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, *descSet, nullptr);
            computeCommandBuffers[i]->dispatch(WIDTH / 16, HEIGHT / 16, 1);
            copyImage(*computeCommandBuffers[i], *outputImage.image, *inputImage.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
            copyImage(*computeCommandBuffers[i], *outputImage.image, swapChainImages[i], vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR);
            computeCommandBuffers[i]->end();
        }
    }

    void copyImage(vk::CommandBuffer& cmdBuf, vk::Image& src, vk::Image& dst, vk::ImageLayout srcLayout, vk::ImageLayout dstLayout)
    {
        using vkIL = vk::ImageLayout;
        transitionImageLayout(cmdBuf, src, vkIL::eUndefined, vkIL::eTransferSrcOptimal);
        transitionImageLayout(cmdBuf, dst, vkIL::eUndefined, vkIL::eTransferDstOptimal);

        vk::ImageCopy copyRegion{};
        copyRegion.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setExtent({ WIDTH, HEIGHT, 1 });
        cmdBuf.copyImage(src, vkIL::eTransferSrcOptimal, dst, vkIL::eTransferDstOptimal, copyRegion);

        transitionImageLayout(cmdBuf, src, vkIL::eTransferSrcOptimal, srcLayout);
        transitionImageLayout(cmdBuf, dst, vkIL::eTransferDstOptimal, dstLayout);
    }

    void allocateDrawCommandBuffers()
    {
        computeCommandBuffers = device->allocateCommandBuffersUnique(
            vk::CommandBufferAllocateInfo{}
            .setCommandPool(*commandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(swapChainImages.size()));
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size());

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            imageAvailableSemaphores[i] = device->createSemaphoreUnique({});
            renderFinishedSemaphores[i] = device->createSemaphoreUnique({});
            inFlightFences[i] = device->createFence({ vk::FenceCreateFlagBits::eSignaled });
        }
    }

    void drawFrame()
    {
        device->waitForFences(inFlightFences[currentFrame], true, UINT64_MAX);

        uint32_t imageIndex = acquireNextImageIndex();

        // Wait for fence
        if (imagesInFlight[imageIndex]) {
            device->waitForFences(imagesInFlight[imageIndex], true, UINT64_MAX);
        }
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];
        device->resetFences(inFlightFences[currentFrame]);

        // Submit draw command
        vk::PipelineStageFlags waitStage{ vk::PipelineStageFlagBits::eComputeShader };
        computeQueue.submit(
            vk::SubmitInfo{}
            .setWaitSemaphores(*imageAvailableSemaphores[currentFrame])
            .setWaitDstStageMask(waitStage)
            .setCommandBuffers(*computeCommandBuffers[imageIndex])
            .setSignalSemaphores(*renderFinishedSemaphores[currentFrame]),
            inFlightFences[currentFrame]);

        // Present image
        presentQueue.presentKHR(
            vk::PresentInfoKHR{}
            .setWaitSemaphores(*renderFinishedSemaphores[currentFrame])
            .setSwapchains(*swapChain)
            .setImageIndices(imageIndex));

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    uint32_t acquireNextImageIndex()
    {
        auto res = device->acquireNextImageKHR(*swapChain, UINT64_MAX, *imageAvailableSemaphores[currentFrame]);
        if (res.result == vk::Result::eSuccess) {
            return res.value;
        }
        throw std::runtime_error("failed to acquire next image!");
    }
};

int main()
{
    Application app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
