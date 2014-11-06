<?php

class Operaciones extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'operaciones';
	}

	public function rules()
	{
		return array(
			array('descripcion', 'length', 'max'=>128),
			array('id, descripcion', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'autorizaciones' => array(self::HAS_MANY, 'Autorizaciones', 'operacion'),
			'eventoses' => array(self::HAS_MANY, 'Eventos', 'operacion'),
			'permisoses' => array(self::HAS_MANY, 'Permisos', 'operacion'),
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
			'descripcion' => Yii::t('app', 'Descripcion'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('descripcion',$this->descripcion,true);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
