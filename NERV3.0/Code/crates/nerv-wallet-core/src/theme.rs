//! The design system (erratum 201): a single theme shared by all four
//! platforms. GitHub's dark palette — familiar, professional, elegant.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Theme {
    pub background: [u8; 3],
    pub surface: [u8; 3],
    pub border: [u8; 3],
    pub text: [u8; 3],
    pub text_muted: [u8; 3],
    pub accent: [u8; 3],
    pub success: [u8; 3],
    pub warning: [u8; 3],
    pub error: [u8; 3],
}

impl Default for Theme {
    fn default() -> Self {
        Theme::dark()
    }
}

impl Theme {
    pub const fn dark() -> Theme {
        Theme {
            background: [0x0D, 0x11, 0x17],
            surface: [0x16, 0x1B, 0x22],
            border: [0x30, 0x36, 0x3D],
            text: [0xE6, 0xED, 0xF3],
            text_muted: [0x8B, 0x94, 0x9E],
            accent: [0x58, 0xA6, 0xFF],
            success: [0x3F, 0xB9, 0x50],
            warning: [0xD2, 0x99, 0x22],
            error: [0xF8, 0x51, 0x49],
        }
    }

    pub const fn light() -> Theme {
        Theme {
            background: [0xFF, 0xFF, 0xFF],
            surface: [0xF6, 0xF8, 0xFA],
            border: [0xD0, 0xD7, 0xDE],
            text: [0x1F, 0x23, 0x28],
            text_muted: [0x65, 0x6D, 0x76],
            accent: [0x09, 0x69, 0xDA],
            success: [0x1A, 0x7F, 0x37],
            warning: [0x9A, 0x67, 0x00],
            error: [0xCF, 0x22, 0x2E],
        }
    }

    pub fn hex(c: [u8; 3]) -> String {
        format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2])
    }
}
