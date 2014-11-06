<?php

/**
 * QRCode Generator
 *
 * @copyright © BryanTan <www.bryantan.info> 2011
 * @license GNU Lesser General Public License v3.0
 * @author Bryan Jayson Tan
 *
 */

include('phpqrcode/qrlib.php');
        
class QRCodeGenerator extends CWidget {

    public $data;
    public $filename = 'qrcode.png';
    public $filePath;
    public $fileUrl;
    public $subfolderVar = false;
    public $subfolderName = 'qrcodes';
    public $errorCorrectionLevel = 'L';
    public $matrixPointSize = 4;
    
    private $fullPath;
    
    public function init()
    {
        if (!isset($this->filePath)){
            $this->filePath = realpath(Yii::app()->getBasePath().'/../uploads');
        }
        
        if(!is_dir($this->filePath)){
            throw new CHttpException(500, "{$this->filePath} does not exists.");
        }else if(!is_writable($this->filePath)){
            throw new CHttpException(500, "{$this->filePath} is not writable.");
        }
        
        if (!isset($this->fileUrl)){
            $this->fileUrl = Yii::app()->baseUrl . '/uploads';
        }
        
        //remember to sanitize user input in real-life solution !!!
        if (!in_array($this->errorCorrectionLevel, array('L','M','Q','H')))
            throw new CException(Yii::t(get_class($this), 'Error Correction Level only accepts L,M,Q,H'));
        
        if (is_null($this->data))
            throw new CException(Yii::t(get_class($this), 'Data must not be empty'));
        
        $this->matrixPointSize = min(max((int)$this->matrixPointSize, 1), 10);
    }
    
    public function run()
    {
        $this->init();
        
        if ($this->subfolderVar){
            $subfolder = $this->filePath.'/' . $this->subfolderName;
            if (!is_dir($subfolder)){
                mkdir($subfolder);
            }
            $this->filePath = $this->filePath . '/' . $this->subfolderName;
            $this->fileUrl = $this->fileUrl . '/' . $this->subfolderName;
        }
        
        $this->filePath = $this->filePath . '/'. $this->filename;
        $this->fullPath = $this->fileUrl . '/'. $this->filename;
        
        QRcode::png($this->data, $this->filePath, $this->errorCorrectionLevel, $this->matrixPointSize, false);
        
        echo CHtml::image($this->fullPath);
    }
}
