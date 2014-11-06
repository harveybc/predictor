<?php

class Reportes extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'reportes';
	}

	public function rules()
	{
		return array(
                        array('Proceso, Area, Equipo','required'), 
			array('Reporte, CFM, COSTO, Corregido', 'numerical', 'integerOnly'=>true),
			array('Presion, Decibeles, ZI, OT, Tamano', 'numerical'),
			array('Path', 'length', 'max'=>255),
                        array('Estado', 'numerical', 'integerOnly'=>true),
                        array('Observaciones', 'length', 'max'=>250),
			array('Proceso, Area, Equipo, Analista, Gas', 'length', 'max'=>50),
			array('Descripcion, Fecha', 'safe'),
			array(' OT, Estado, Observaciones, id, Reporte, Path, Presion, Decibeles, Descripcion, ZI, Proceso, Area, Equipo, Analista, OT, Fecha, Gas, Tamano, CFM, COSTO, Corregido', 'safe', 'on'=>'search'),
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
			'Reporte' => Yii::t('app', 'Reporte'),
			'Path' => Yii::t('app', 'Imagen'),
			'Presion' => Yii::t('app', 'Presión'),
			'Decibeles' => Yii::t('app', 'Decibeles'),
			'Descripcion' => Yii::t('app', 'Descripción'),
			'ZI' => Yii::t('app', 'Aviso ZI'),
			'Proceso' => Yii::t('app', 'Area'),
			'Area' => Yii::t('app', 'Proceso'),
			'Equipo' => Yii::t('app', 'Equipo'),
			'Analista' => Yii::t('app', 'Analista'),
			'OT' => Yii::t('app','Orden de Trabajo'),
			'Fecha' => Yii::t('app', 'Fecha'),
			'Gas' => Yii::t('app', 'Gas'),
			'Tamano' => Yii::t('app', 'Tamano'),
			'CFM' => Yii::t('app', 'Cfm'),
			'COSTO' => Yii::t('app', 'Costo'),
                        'Estado' => Yii::t('app', 'Estado'),
                        'Observaciones' => Yii::t('app', 'Observaciones'),            
			'Corregido' => Yii::t('app', 'Corregido'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);
                
		//$criteria->compare('Reporte',$this->Reporte);

		//$criteria->compare('Path',$this->Path,true);

		//$criteria->compare('Presion',$this->Presion);

		//$criteria->compare('Decibeles',$this->Decibeles);

		$criteria->compare('Descripcion',$this->Descripcion,true);

		$criteria->compare('ZI',$this->ZI);

		$criteria->compare('Proceso',$this->Proceso,true);

		$criteria->compare('Area',$this->Area,true);

		$criteria->compare('Equipo',$this->Equipo,true);

		$criteria->compare('Analista',$this->Analista,true);

		$criteria->compare('OT',$this->OT);

		$criteria->compare('Fecha',$this->Fecha,true);

                $criteria->compare('Estado', $this->Estado);

//		$criteria->compare('Gas',$this->Gas,true);

//		$criteria->compare('Tamano',$this->Tamano);

//		$criteria->compare('CFM',$this->CFM);

		$criteria->compare('COSTO',$this->COSTO);

		$criteria->compare('Corregido',$this->Corregido);
		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}

    public function searchEquipos($area, $equipo)
        {
            $criteria = new CDbCriteria;
            $criteria->compare('Area',
                               $area,
                               true);
            $criteria->compare('Equipo',
                               $equipo,
                               true);
		$criteria->compare('id',$this->id,true);
                
		//$criteria->compare('Reporte',$this->Reporte);

		//$criteria->compare('Path',$this->Path,true);

		$criteria->compare('Presion',$this->Presion);

		$criteria->compare('Decibeles',$this->Decibeles);

		$criteria->compare('Descripcion',$this->Descripcion);

//		$criteria->compare('ZI',$this->ZI);

		$criteria->compare('Proceso',$this->Proceso);

		$criteria->compare('Analista',$this->Analista);

		$criteria->compare('OT',$this->OT);

		$criteria->compare('Fecha',$this->Fecha);

//		$criteria->compare('Gas',$this->Gas,true);

//		$criteria->compare('Tamano',$this->Tamano);

//		$criteria->compare('CFM',$this->CFM);

		$criteria->compare('COSTO',$this->COSTO);
            
            return new CActiveDataProvider(get_class($this), array (
                'criteria' => $criteria,
            ));
            /*
            $sql = 'SELECT * FROM motores WHERE Equipo="' . $equipo . '" AND Area="'.$area.'" ORDER BY Equipo';
            $myDataProvider = new CSqlDataProvider($sql, array (
                    'keyField' => 'id',
                    'pagination' => array (
                        'pageSize' => 10,
                    ),
                    )
            );
            return $myDataProvider;
             * 
             */
        }
        
    public function searchEquiposArea($area)
        {
                        $criteria = new CDbCriteria;
            $criteria->compare('Area',
                               $area,
                               false);
            
            		$criteria->compare('Presion',$this->Presion);

		$criteria->compare('Decibeles',$this->Decibeles);

		$criteria->compare('Descripcion',$this->Descripcion);

//		$criteria->compare('ZI',$this->ZI);

		$criteria->compare('Proceso',$this->Proceso);

		$criteria->compare('Analista',$this->Analista);

		$criteria->compare('OT',$this->OT);

		$criteria->compare('Fecha',$this->Fecha);

//		$criteria->compare('Gas',$this->Gas,true);

//		$criteria->compare('Tamano',$this->Tamano);

//		$criteria->compare('CFM',$this->CFM);

		$criteria->compare('COSTO',$this->COSTO);
            
            return new CActiveDataProvider(get_class($this), array (
                'criteria' => $criteria,
            ));
            /*
            $sql = 'SELECT * FROM motores WHERE Equipo="' . $equipo . '" AND Area="'.$area.'" ORDER BY Equipo';
            $myDataProvider = new CSqlDataProvider($sql, array (
                    'keyField' => 'id',
                    'pagination' => array (
                        'pageSize' => 10,
                    ),
                    )
            );
            return $myDataProvider;
             * 
             */
        }

}
