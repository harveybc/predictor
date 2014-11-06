<?php

class Autorizaciones extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'autorizaciones';
	}

	public function rules()
	{
		return array(
			array('usuario, documento, operacion, autorizado', 'numerical', 'integerOnly'=>true),
			array('id, usuario, documento, operacion, autorizado', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'usuario0' => array(self::BELONGS_TO, 'Usuarios', 'usuario'),
			'documento0' => array(self::BELONGS_TO, 'Documentos', 'documento'),
			'operacion0' => array(self::BELONGS_TO, 'Operaciones', 'operacion'),
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
			'documento' => Yii::t('app', 'Documento'),
			'operacion' => Yii::t('app', 'Operacion'),
			'autorizado' => Yii::t('app', 'Autorizado'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('usuario',$this->usuario);

		$criteria->compare('documento',$this->documento);

		$criteria->compare('operacion',$this->operacion);

		$criteria->compare('autorizado',$this->autorizado);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
