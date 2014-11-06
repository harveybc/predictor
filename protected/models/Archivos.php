<?php

class Archivos extends CActiveRecord
{
    /* PROPERTY FOR RECEIVING THE FILE FROM FORM*/          
public $nombre;
/* MUST BE THE SAME AS FORM INPUT FIELD NAME*/          

/**
*saves the name, size ,type and data of the uploaded file
*/
public function beforeSave()
{
if($file=CUploadedFile::getInstance($this,'nombre'))
{

$this->nombre=$file->name;
$this->tipo=$file->type;
$this->tamano=$file->size;
$this->contenido=file_get_contents($file->tempName);

}

return parent::beforeSave();;

}
    
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'archivos';
	}

	public function rules()
	{
		return array(
			//array('nombre, tipo, tamano, contenido', 'required'),
			array('nombre', 'length', 'max'=>512),
			array('tipo', 'length', 'max'=>64),
			array('tamano', 'length', 'max'=>20),
			array('id, nombre, tipo, tamano, contenido', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
		);
	}

	public function behaviors()
	{
		return array('CAdvancedArBehavior',
				array('class' => 'ext.CAdvancedArBehavior')
				);
	}

	public function attributeLabels()
	{
		return array(
			'id' => Yii::t('app', 'ID'),
			'nombre' => Yii::t('app', 'Archivo'),
			'tipo' => Yii::t('app', 'Tipo'),
			'tamano' => Yii::t('app', 'Tamano'),
			'contenido' => Yii::t('app', 'Contenido'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('nombre',$this->nombre,true);

		$criteria->compare('tipo',$this->tipo,true);

		$criteria->compare('tamano',$this->tamano,true);

		$criteria->compare('contenido',$this->contenido,true);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
