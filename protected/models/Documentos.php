<?php

class Documentos extends CActiveRecord
{
    
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'documentos';
	}

	public function rules()
	{
		return array(
			array('descripcion,permitirAdiciones, permitirAnotaciones, autorizarOtros, requiereAutorizacion,', 'required'),
			array('permitirAdiciones, permitirAnotaciones, autorizarOtros, requiereAutorizacion, secuencia, ordenSecuencia, eliminado, conservacionPermanente', 'numerical', 'integerOnly'=>true),
			array('descripcion', 'length', 'max'=>128),
                        //array('descripcion','unique'),
			array('conservacionInicio, conservacionFin', 'safe'),
			array('id, descripcion, permitirAdiciones, permitirAnotaciones, autorizarOtros, requiereAutorizacion, secuencia, ordenSecuencia, eliminado, conservacionInicio, conservacionFin, conservacionPermanente', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'anotaciones' => array(self::HAS_MANY, 'Anotaciones', 'documento'),
			'autorizaciones' => array(self::HAS_MANY, 'Autorizaciones', 'documento'),
			'competenciasDocumentoses' => array(self::HAS_MANY, 'CompetenciasDocumentos', 'documento'),
			'secuencia0' => array(self::BELONGS_TO, 'Secuencias', 'secuencia'),
			'ordenSecuencia0' => array(self::BELONGS_TO, 'OrdenSecuencias', 'ordenSecuencia'),
			'metaDocs' => array(self::HAS_MANY, 'MetaDocs', 'documento'),
			'permisoses' => array(self::HAS_MANY, 'Permisos', 'documento'),
			'tags' => array(self::HAS_MANY, 'Tags', 'documento'),
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
			'descripcion' => Yii::t('app', 'Título'),
			'permitirAdiciones' => Yii::t('app', 'Permitir Adiciones'),
			'permitirAnotaciones' => Yii::t('app', 'Permitir Anotaciones'),
			'autorizarOtros' => Yii::t('app', 'Autorizar Otros'),
			'requiereAutorizacion' => Yii::t('app', 'Requiere Autorización'),
			'secuencia' => Yii::t('app', 'Secuencia'),
			'ordenSecuencia' => Yii::t('app', 'Orden Secuencia'),
			'eliminado' => Yii::t('app', 'Eliminado'),
			'conservacionInicio' => Yii::t('app', 'Conservación Inicio'),
			'conservacionFin' => Yii::t('app', 'Conservación Fin'),
			'conservacionPermanente' => Yii::t('app', 'Conserv. Permanente'),
			
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('descripcion',$this->descripcion,true);

		$criteria->compare('permitirAdiciones',$this->permitirAdiciones);

		$criteria->compare('permitirAnotaciones',$this->permitirAnotaciones);

		$criteria->compare('autorizarOtros',$this->autorizarOtros);

		$criteria->compare('requiereAutorizacion',$this->requiereAutorizacion);

		$criteria->compare('secuencia',$this->secuencia);

		$criteria->compare('ordenSecuencia',$this->ordenSecuencia);

		$criteria->compare('eliminado',$this->eliminado);

		$criteria->compare('conservacionInicio',$this->conservacionInicio,true);

		$criteria->compare('conservacionFin',$this->conservacionFin,true);

		$criteria->compare('conservacionPermanente',$this->conservacionPermanente);

	

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
