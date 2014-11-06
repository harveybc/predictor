<?php

class Anotaciones extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'anotaciones';
	}

	public function rules()
	{
		return array(
			array('contenido', 'required'),
			array('usuario, documento, eliminado', 'numerical', 'integerOnly'=>true),
			array('descripcion', 'length', 'max'=>256),
			array('id, usuario, descripcion, documento, eliminado,contenido', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'usuario0' => array(self::BELONGS_TO, 'Usuarios', 'usuario'),
			'documento0' => array(self::BELONGS_TO, 'MetaDocs', 'documento'),
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
			'usuario' => Yii::t('app', 'Usuario'),
			'descripcion' => Yii::t('app', 'Descripcion'),
			'documento' => Yii::t('app', 'Documento'),
			'eliminado' => Yii::t('app', 'Eliminado'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('usuario',$this->usuario);

		$criteria->compare('descripcion',$this->descripcion,true);

		$criteria->compare('documento',$this->documento);

		$criteria->compare('eliminado',$this->eliminado);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
