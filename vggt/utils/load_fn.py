# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np


def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.
    [中文翻译]
    加载并预处理图像，通过居中填白以使其成为正方形，并调整大小为目标尺寸。
    同时也会返回变换后原始像素的具体位置信息。

    Args:
        image_path_list (list): List of paths to image files
                                [中文翻译] 图像文件路径的列表
        target_size (int, optional): Target size for both width and height. Defaults to 518.
                                     [中文翻译] 对应目标尺寸的宽和高。注释中说明默认值为 518（注：函数签名中目前默认值为 1024）。

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
                          [中文翻译] 预处理后图像的批量张量，形状为 (N, 3, 目标大小, 目标大小)
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
                          [中文翻译] 针对每张图包含原图对应参数 [x1, y1, x2, y2, width, height] 的数组（注：实际生成的张量有 6 个维度而非 5 个）
        )

    Raises:
        ValueError: If the input list is empty
                    [中文翻译] 如果输入的图片路径列表为空则引发该错误
    """
    # Check for empty list
    # [中文翻译] 检查列表是否为空
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    # [中文翻译] 将变量名为 position_info 重命名成 original_coords，使其更具表达性（记录原图像坐标）
    to_tensor = TF.ToTensor() # 初始化：用于将 PIL 图像实体转换为 PyTorch 的张量并且把数值 [0~255] 归一化为 [0.0~1.0]

    for image_path in image_path_list:
        # Open image
        # [中文翻译] 打开图像
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        # [中文翻译] 如果图像自带透明 alpha 通道的话，就直接将其混叠添加在一个纯白色背景里
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        # [中文翻译] 将图像转换到统一 RGB 格式
        img = img.convert("RGB")

        # Get original dimensions
        # [中文翻译] 获取原始宽、高尺寸
        width, height = img.size

        # Make the image square by padding the shorter dimension
        # [中文翻译] 以宽、高中的最长一边作为基准大小从而在随后的短边进行空白填充以此保证原图成为长宽等边的正方形
        max_dim = max(width, height)

        # Calculate padding
        # [中文翻译] 计算在左边和顶部需要具体做填充补偿处理的空余留位参数大小
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        # [中文翻译] 专门算出重新调整大小情况下的缩放级别与比率
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        # [中文翻译] 开始运算原图像部分最终缩减成目标尺寸空间过程当里的对应真实位置坐标标定
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        # [中文翻译] 下单原先那张图像有关各种详细情况如四面八方的定位数值同初始宽高速览值等予以寄存到列表内
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        # [中文翻译] 创造性地新开辟出一片纯黑正方形空场地，并把最开始的原画面照本宣科依据计算出的距离偏差做定点粘贴放置好
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        # [中文翻译] 利用双三次插值降维超分采样等比例重新放大或削微缩小至规范目标的格式和尺寸
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        # [中文翻译] 最终转换为 Torch 处理所需 Tensor 字典集合类型存内
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    # [中文翻译] 按照指定排列轴向批量把这一整摞的所有数据堆叠绑定在一起变成带有整齐 Batch Number 的形态
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    # [中文翻译] 考虑当仅仅处理只含单张照片任务时候增设外部补位数据特征阶层并力求维持形状规整规范没有缺陷失控现象即 [1, C, H, W]
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.
    [中文翻译]
    一个快速准备及预处理针对模型输入图像的便捷函数接口。
    此处默认并假设处理这所有的相关图像最后将会保持具备相同统一规定的轮廓与大小，那样才能相对容易进行捆绑合批次(batching)输入执行运算；不过咱们目前的这套模型设计也恰好具备足够泛化的兼容性能力去照样非常稳定接纳各式花样的不用统一限制的具体尺寸输入条件。

    Args:
        image_path_list (list): List of paths to image files
                                [中文翻译] 提供了含有所有需要引用的真实图像档案的定位存储地址系列清册
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                              [中文翻译] 指代处理过程中前置筛选与定制剪辑策略形式，必须在 "crop" 裁剪 或者 "pad" 补白之中做出择一决定。
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                               [中文翻译] "crop"（系统内置第一顺位候选项）：强制定制化要求设定图像全宽为严格要求条件下的具体数值 518px；随后看其如果多超出的纵向高度部分就会采用按圆心基准线上下居中等量做裁切手段处理之。
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.
                               [中文翻译] "pad"：该手法意欲最大程度把原来那些旧像素资产均等完全地保留下。通过保证其当前的最大尺寸变达到标准要求 518px 并开始给另一较小型长轴追加空白区域边直到将全局都硬生生拼凑达到呈现绝对正方形框形态要求。

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)
                      [中文翻译] 返回处理后已被打包的综合张量组（N个样本集, 共合着通道占有为3, 总高度为 H，全宽度占有为 W）

    Raises:
        ValueError: If the input list is empty or if mode is invalid
                    [中文翻译] 引发该错误说明若发现使用者发来的原始载入条目录属于完全空白没任何内容的情况以及发现它强指定的这参数模式不对压根非法的时候报错提醒中止动作

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
          [中文翻译] 对于带有大小悬殊差别特征属性的其他单独对象照片来说将会在背景上进行强添加空白纯白布贴条进行衬底弥合（此处的纯白色表现为张量中恒等于1.0）
        - A warning is printed when images have different shapes
          [中文翻译] 当系统侦查鉴别的期间撞上了个别的有些图片形似完全不同就会进行报警系统文本提示输出
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
          [中文翻译] 仅在采取方案 mode="crop" 条件下成立运作期间内：本工作流程首要明确担保这其中图形宽度的精准设定等于标准的518px同时也致力于尽力坚守那固有画面内容表现形式上的画面横竖等距对比比例并针对若是其中出现高度实在大于518px便着手进行大刀阔斧般的从其正版中心向外裁剪。
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
          [中文翻译] 更替至取用方案 mode="pad" 大环境下：此流程就会将大长条或者是宽窄体这些无论那个才是里头真正能够算得上具有所谓首屈一指长维标定物确保限制调整到刚好在上限范围518px下再通过补足稍逊其较小次级别另一短面使其共同形成完美（外径等比为518x518）的大体矩形框形图样。
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
          [中文翻译] 各大维度均需被严丝合缝精心改制必须整改调校为恰能够顺顺当当受14作除法后得出规整计算值的结局标准来从而契合法理上模型内在严格死板的需求特性。
    """
    # Check for empty list
    # [中文翻译] 系统初始先行清察提交上来的表单列表到底当前存不存在为空这一可能状态
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    # [中文翻译] 开始启动检验检查操作行为验证此选择参数合理正确合规情况与否
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set() # 特别提供了一块用来集聚收集目前整理出的所有被调整修正规格化以后得到的各式长高形态分布归集之地（以集合的方式只收集唯一尺寸记录）
    to_tensor = TF.ToTensor() # 将各类外部媒体转换构建为张量形态对象的重要中间手段操作
    target_size = 518 # 规范中明确规划指定并统一要用的那个唯一基本标准化核心基准衡量尺寸518px大小长短

    # First process all images and collect their shapes
    # [中文翻译] 第一道正式关隘起手式就是先遍历一遍全部图档挨个进行规范排查和完成初步修改操作并同时汇总积攒起它自身修正之后的定版大小身材数据
    for image_path in image_path_list:
        # Open image
        # [中文翻译] 读取与正式开启图片
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        # [中文翻译] 面临它万一看上去带有了某些透明化度量的这种层面上附加图层信息的情况则将它们全面平移贴花并叠加到一片白茫茫无垢白色基础画板里：
        if img.mode == "RGBA":
            # Create white background
            # [中文翻译] 亲手新做一块全部素色纯白的透明画板子
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            # [中文翻译] 使用自带叠加能力特性去对着白色大画板子上把原身彻底铺加上去混同覆盖其原原本本的内容
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        # [中文翻译] 此番现在即是要将这等文件再次回归转化回归至传统通用模式的 "RGB" (做这步的最主要的核心目的便是将上一次补进去给它赋透明处加上那道底层纯正白色完全固定下来从而生效定型落实)
        img = img.convert("RGB")

        width, height = img.size # 提取拿到最新的自身准确长宽高详细数据报表

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            # [中文翻译] 开始保持在一直不能改变它原来横长或上下高的真实视角的对应比率的大前提之下面将最大那个轴段生硬定在限制红圈线 518px 前不许越前雷池
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
                # [中文翻译] 特地专门强行把改短了的新计算高度也要进行二次调整修正满足绝对能够恰好地直接让这计算得数刚好可由14完整全分解切去的要求标准规定之中去
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
                # [中文翻译] 特地专门强行把新改变出来的那一点点的修正计算宽度也要进行调整来使其切切当当实打实得确保毫无怨言直接就将得数让数字 14 一刀直挺精准切割掉满足无余数存活要求情况出现才罢休
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            # [中文翻译] 保留从前的最为初始原始直接生硬老派的一流操作规则配置习惯法则：直接将整改尺寸参数的设定范围宽固定在标准刻度线上卡死 518px 大小
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            # [中文翻译] 全程始终必须确保一直尽力坚守原来视觉宽高比例特征不遭受人为损毁原则底边之下去再次依据已确认宽度对此时当下相对应高度重新进行算计并让最终落板出来的成效数值依然完美规矩让标准数字14做等比例绝对干净分解全灭操作无遗漏发生。
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        # [中文翻译] 获取所有经过严苛周密精密打算与谋划计算而获知的全新调整尺寸后立即根据命令执行整体对象双向二次全面重造插值化尺寸变更处理计划
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)
        # [中文翻译] 并且随即将其向张量格式进行根本上属性转轨大迁徙转变(其中所包含各处的所有色彩亮度都会在这当口转眼间尽皆变成了一堆从0.0慢慢渐长过渡直达1.0范畴之间微小刻度浮动物质而已)

        # Center crop height if it's larger than 518 (only in crop mode)
        # [中文翻译] 后续如果在裁切方案 "crop" 操作下因为先前按宽重置后它的那个最新出炉生成实际新高如果很不巧刚好偏大且跑冒越出了 518px 红圈那就干脆二话没多只朝那它的心窝中间下手切掉剩下周围那多余无用碍事的地方就完事了。
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2 # 精密盘算出中心对头裁剪将要开始从多靠上的顶点具体确切的坐标轴起点开刀开始剥除的开端部位处起
            img = img[:, start_y : start_y + target_size, :] # [C, H, W] 对当前目标直接按要求切割剔出无用之物只留符合规定那部分精髓骨血核心地段。

        # For pad mode, pad to make a square of target_size x target_size
        # [中文翻译] 但那反面来讲在补充 "pad" 行进过程中若是没有那条件那就干脆靠往里塞边衬空贴画让其生生自己填充长得大成那个目标中那完美方形标准规格(Target_size乘Target_size)大尺寸就行咯。
        if mode == "pad":
            h_padding = target_size - img.shape[1] # 先摸查到底缺少多大短小的欠失所需的高度尺寸多少差事
            w_padding = target_size - img.shape[2] # 再清点算好这缺欠下的左右需补白空白缺漏尺寸宽差距

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                # [中文翻译] 即刻起手用最鲜亮光洁一抹纯净洁白白色物质对图象缺隙开始周至细致添料去作白底边沿外拓展补白工作行径(这在表示其中张量里的纯洁雪白表示为其数据值为一准儿等于是1.0恒定数据项参数), 在此处函数对于周界执行填空扩展要求排位依序明确指定参数应当首重从先在(左端开始,其后依次按着右端,然后再顶层上部,最底层下端)这么依次按排处理过去即可
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    # [中文翻译] 最后一步来复盘重新再检视确认一下我们这里弄出来的这么多张经过历经风霜打磨调整修补后是否还会出现了存在各式依然参差不一样不同款式尺寸相伴而生形形色色这类的尴尬窘况发生。
    # 按照此前那纸上谈兵设计好构建起的那最初一板一眼书面上写着的原本最为崇高设计底层逻辑其实咱家当前使用的模型本应在针对去面临及兼容适配哪怕遇到多有极其古怪或甚至尺寸相异参差不同之各等杂乱情况时理所当然理论同样本也应该当是属于那种无视所有一切阻碍始终能够保持其相当完美且不惧应对那些变化能力发挥卓越效果的存在啊。
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        # [中文翻译] 随即着手从中这一堆各式繁杂图像信息群类中间仔细摸寻并找见筛选识别挑出其整体批次内部现下包含有着最高及最为之巨大化尺长外廓边线边沿的最大极点高程同宏大极远开阔之长阔处在哪端所在(最大维度找寻)。
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        # [中文翻译] 把刚那从这里头寻觅扒出的找来的这一众长大小不齐者只要遇上有那小了个别实在有必要或不够条件情况就干脆将其也进行这最末后手段执行进行强制扩展添白外边缘等最终补齐至最大齐边规范统一行径步骤
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # 当有图片不达标最豪气尺寸大小那就在边边上帮衬它外补点纯厚白彩给填成合规矩
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images
    # [中文翻译] 统帅号令让前面全经历这诸般历行之后得来成果们集中统一做拼接整合为单一个有着多层排列 Batch 号属性等格式样板之终极版特化形态输出 [N, C, H, W]

    # Ensure correct shape when single image
    # [中文翻译] 为这极个别仅剩独自跑单存在一张独立存在执行运作下确保也能让其一直维持与保持大队伍行列格式同质化及规格正确体例做保证。
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        # [中文翻译] 只去针对排查跟仔细审验查证看看这个情况那尺寸最终长出状态是到底符不符带有那以代表自身就是代表整整一整个批次 1 位代表性指标属性情况规范否 (1, C, H, W) 
        if images.dim() == 3:
            images = images.unsqueeze(0) # 将它第一位硬行加上扩展补填以撑大这一序列使刚好契合 Batch 最基本规定原则机制规范情况

    return images
